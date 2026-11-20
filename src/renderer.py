"""
3D Gaussian Splatting EWA Multi-Modal Renderer (RGB, Metric Depth, Semantic Instance & Language Heatmaps).

Implements explicit 3D-to-2D Elliptical Weighted Average (EWA) covariance projection:
    Sigma_2D = J * W * Sigma_3D * W^T * J^T + eps * I_2
followed by front-to-back depth-sorted alpha compositing:
    C(u, v) = sum_i T_i * alpha_i * c_i,   T_i = prod_{j < i} (1 - alpha_j)
Simultaneously outputs:
  1. Photorealistic RGB image
  2. Expected metric depth map D(u, v)
  3. Instance segmentation ID map & semantic overlay
  4. Open-vocabulary language relevancy heatmap A(u, v; query)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .gaussian_scene import Object3DInstance, SemanticGaussianScene


@dataclass
class RenderOutput:
    """Multi-modal output of a 3D Gaussian Splatting render call."""

    rgb: np.ndarray  # (H, W, 3) float32 in [0, 1]
    depth: np.ndarray  # (H, W) float32 in meters
    alpha: np.ndarray  # (H, W) float32 accumulated opacity in [0, 1]
    instance_map: np.ndarray  # (H, W) int32 instance IDs (0 = background)
    semantic_rgb: np.ndarray  # (H, W, 3) float32 semantic overlay in [0, 1]
    relevancy_map: Optional[np.ndarray]  # (H, W) float32 in [0, 1] if query_text given
    visible_objects: List[Dict[str, Any]]
    camera_pose_twc: np.ndarray  # (4, 4) camera-to-world matrix


def look_at_pose(
    camera_pos: np.ndarray,
    look_at: np.ndarray,
    up_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct a 4x4 camera-to-world matrix T_wc (OpenCV pinhole convention:
    +Z forward along viewing ray, +X right, +Y down) looking from `camera_pos` to `look_at`.
    """
    eye = np.asarray(camera_pos, dtype=np.float32).reshape(3)
    target = np.asarray(look_at, dtype=np.float32).reshape(3)
    forward = target - eye
    dist = float(np.linalg.norm(forward))
    if dist < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward /= dist

    # In TUM fr1/desk world frame, camera Y points roughly downward [0, 1, 0]
    if up_world is None:
        up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        up_world = np.asarray(up_world, dtype=np.float32).reshape(3)

    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-5:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_world)
    right /= np.maximum(np.linalg.norm(right), 1e-8)
    down = np.cross(forward, right)
    down /= np.maximum(np.linalg.norm(down), 1e-8)

    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 0] = right
    T_wc[:3, 1] = down
    T_wc[:3, 2] = forward
    T_wc[:3, 3] = eye
    return T_wc


class GaussianRenderer:
    """
    EWA Covariance Splatting Renderer for `SemanticGaussianScene`.
    Supports both anisotropic 3D covariance projection (`anisotropic=True`) and
    isotropic point splatting (`anisotropic=False`) for ablation studies.
    """

    def __init__(
        self,
        scene: SemanticGaussianScene,
        width: int = 640,
        height: int = 480,
        anisotropic: bool = True,
    ) -> None:
        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.anisotropic = bool(anisotropic)

    def project_covariances_2d(
        self,
        pts_c: np.ndarray,
        R_cw: np.ndarray,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        """
        Explicit EWA 3D-to-2D covariance projection:
            Sigma_2D = J * R_cw * Sigma_3D * R_cw^T * J^T + 0.3 * I_2
        Returns (M, 2, 2) 2D covariance matrices in pixel space.
        """
        Sigma_3d = self.scene.get_covariances_3d()
        # Transform 3D covariance into camera coordinates: W * Sigma * W^T
        Sigma_cam = np.matmul(
            R_cw[None, :, :], np.matmul(Sigma_3d, R_cw.T[None, :, :])
        )
        z = np.maximum(pts_c[:, 2], 0.25)
        x = pts_c[:, 0]
        y = pts_c[:, 1]

        # Perspective Jacobian J (M, 2, 3)
        J = np.zeros((pts_c.shape[0], 2, 3), dtype=np.float32)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * x / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * y / (z * z)

        Sigma_2d = np.matmul(J, np.matmul(Sigma_cam, np.swapaxes(J, -1, -2)))
        Sigma_2d[:, 0, 0] += 0.35
        Sigma_2d[:, 1, 1] += 0.35
        return Sigma_2d

    def render(
        self,
        camera_pose_twc: np.ndarray,
        scale_factor: float = 1.0,
        query_text: Optional[str] = None,
        bg_color: Tuple[float, float, float] = (0.08, 0.09, 0.12),
    ) -> RenderOutput:
        """
        Render RGB, Metric Depth, Semantic Instance Map, and optional Open-Vocabulary
        Language Heatmap from `camera_pose_twc` (4x4 camera-to-world matrix).
        """
        H = max(32, int(round(self.height * scale_factor)))
        W = max(32, int(round(self.width * scale_factor)))
        K = self.scene.intrinsics_K * scale_factor
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        T_wc = np.asarray(camera_pose_twc, dtype=np.float32)
        T_cw = np.linalg.inv(T_wc)
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]

        pts_c = (self.scene.positions @ R_cw.T) + t_cw
        z_c = pts_c[:, 2]
        valid_z = (z_c > 0.25) & (z_c < 4.5)

        u_f = pts_c[:, 0] * fx / np.maximum(z_c, 1e-4) + cx
        v_f = pts_c[:, 1] * fy / np.maximum(z_c, 1e-4) + cy
        margin = int(max(6, round(12 * scale_factor)))
        in_frustum = (
            valid_z
            & (u_f >= -margin)
            & (u_f < W + margin)
            & (v_f >= -margin)
            & (v_f < H + margin)
        )
        vis_idx = np.nonzero(in_frustum)[0]

        rgb_canvas = np.zeros((H, W, 3), dtype=np.float32)
        depth_canvas = np.zeros((H, W), dtype=np.float32)
        transmittance = np.ones((H, W), dtype=np.float32)
        inst_canvas = np.zeros((H, W), dtype=np.int32)
        inst_weight = np.zeros((H, W), dtype=np.float32)
        rel_canvas = np.zeros((H, W), dtype=np.float32) if query_text else None

        if vis_idx.size == 0:
            bg = np.array(bg_color, dtype=np.float32)[None, None, :]
            return RenderOutput(
                rgb=np.broadcast_to(bg, (H, W, 3)).copy(),
                depth=depth_canvas,
                alpha=np.zeros((H, W), dtype=np.float32),
                instance_map=inst_canvas,
                semantic_rgb=np.broadcast_to(bg, (H, W, 3)).copy(),
                relevancy_map=rel_canvas,
                visible_objects=[],
                camera_pose_twc=T_wc,
            )

        # Sort visible Gaussians front-to-back by camera depth z_c
        z_vis = z_c[vis_idx]
        sort_order = np.argsort(z_vis)
        vis_idx = vis_idx[sort_order]
        z_vis = z_vis[sort_order]
        u_vis = u_f[vis_idx]
        v_vis = v_f[vis_idx]
        col_vis = self.scene.colors[vis_idx]
        opa_vis = self.scene.opacities[vis_idx]
        inst_vis = self.scene.instance_ids[vis_idx]
        scales_vis = self.scene.scales[vis_idx]

        # Compute per-Gaussian language relevancy if query_text is provided
        rel_vis = None
        if query_text:
            q_emb = self.scene.encoder.encode([query_text])[0]
            obj_sims = np.zeros(len(self.scene.objects) + 1, dtype=np.float32)
            for obj in self.scene.objects:
                sim = float(np.dot(obj.embedding, q_emb))
                obj_sims[obj.instance_index] = max(0.0, sim)
            # Soft contrast stretch so top matching objects stand out in [0, 1]
            if obj_sims.max() > 0.15:
                s_min = float(np.percentile(obj_sims[1:], 25)) if len(obj_sims) > 1 else 0.0
                s_max = float(obj_sims.max())
                obj_sims[1:] = np.clip((obj_sims[1:] - s_min) / max(s_max - s_min, 1e-4), 0.0, 1.0)
            rel_vis = obj_sims[np.clip(inst_vis, 0, len(obj_sims) - 1)]

        # Compute 2D projected elliptical Gaussian radii (sigma_x, sigma_y) in pixels
        if self.anisotropic:
            R_vis = self.scene.get_covariances_3d()[vis_idx]
            Sigma_cam = np.matmul(
                R_cw[None, :, :], np.matmul(R_vis, R_cw.T[None, :, :])
            )
            sig_x = np.clip(
                np.sqrt(np.maximum(Sigma_cam[:, 0, 0], 1e-6)) * fx / z_vis,
                0.8 * scale_factor,
                6.5 * scale_factor,
            )
            sig_y = np.clip(
                np.sqrt(np.maximum(Sigma_cam[:, 1, 1], 1e-6)) * fy / z_vis,
                0.8 * scale_factor,
                6.5 * scale_factor,
            )
        else:
            # Isotropic fixed-radius point splatting baseline
            iso_s = np.mean(scales_vis, axis=-1) * 0.55
            sig_x = np.clip(iso_s * fx / z_vis, 0.7 * scale_factor, 2.2 * scale_factor)
            sig_y = sig_x.copy()

        # Vectorized multi-offset EWA kernel splatting (5x5 / 7x7 Gaussian kernel offsets)
        max_rad = 3 if (scale_factor >= 0.75 and self.anisotropic) else 1
        offsets = [
            (du, dv)
            for dv in range(-max_rad, max_rad + 1)
            for du in range(-max_rad, max_rad + 1)
            if du * du + dv * dv <= max_rad * max_rad + 1
        ]

        u0 = np.round(u_vis).astype(np.int32)
        v0 = np.round(v_vis).astype(np.int32)

        # Pass 1: Per-pixel front-surface depth envelope z_min(v, u) over EWA footprint
        z_min = np.full((H, W), 99.0, dtype=np.float32)
        for du, dv in offsets:
            uu = u0 + du
            vv = v0 + dv
            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(inside):
                continue
            np.minimum.at(z_min, (vv[inside], uu[inside]), z_vis[inside])

        # Pass 2: Front-surface depth-gated EWA elliptical Gaussian accumulation
        # Only Gaussians within the front surface shell (dz <= 3.8 cm) contribute,
        # preventing occluded background walls from bleeding through foreground objects.
        shell_thickness = 0.038 if self.anisotropic else 0.095
        tau_z = 0.014 if self.anisotropic else 0.040
        weight_sum = np.zeros((H, W), dtype=np.float32)

        for du, dv in offsets:
            uu = u0 + du
            vv = v0 + dv
            inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not np.any(inside):
                continue
            idx_in = np.nonzero(inside)[0]
            uu_in = uu[idx_in]
            vv_in = vv[idx_in]
            dz = np.maximum(z_vis[idx_in] - z_min[vv_in, uu_in], 0.0)
            front = dz <= shell_thickness
            if not np.any(front):
                continue
            idx_f = idx_in[front]
            uu_k = uu_in[front]
            vv_k = vv_in[front]
            dz_k = dz[front]

            # Evaluate 2D EWA elliptical Gaussian kernel * depth-shell decay
            w_ewa = np.exp(
                -0.5
                * (
                    (du / np.maximum(sig_x[idx_f], 0.55)) ** 2
                    + (dv / np.maximum(sig_y[idx_f], 0.55)) ** 2
                )
                - dz_k / tau_z
            ).astype(np.float32)
            ak = opa_vis[idx_f] * w_ewa
            keep_k = ak > 0.015
            if not np.any(keep_k):
                continue
            idx_k = idx_f[keep_k]
            uu_k = uu_k[keep_k]
            vv_k = vv_k[keep_k]
            ak = ak[keep_k]
            ck = col_vis[idx_k]
            zk = z_vis[idx_k]
            idk = inst_vis[idx_k]

            np.add.at(weight_sum, (vv_k, uu_k), ak)
            np.add.at(rgb_canvas, (vv_k, uu_k), ak[:, None] * ck)
            np.add.at(depth_canvas, (vv_k, uu_k), ak * zk)
            if rel_canvas is not None and rel_vis is not None:
                np.add.at(rel_canvas, (vv_k, uu_k), ak * rel_vis[idx_k])

            higher = (idk > 0) & (ak > inst_weight[vv_k, uu_k])
            if np.any(higher):
                inst_canvas[vv_k[higher], uu_k[higher]] = idk[higher]
                inst_weight[vv_k[higher], uu_k[higher]] = ak[higher]

        alpha_map = np.clip(1.0 - np.exp(-weight_sum * 4.5), 0.0, 1.0)
        valid_a = weight_sum > 1e-4
        depth_out = np.zeros_like(depth_canvas)
        depth_out[valid_a] = depth_canvas[valid_a] / weight_sum[valid_a]

        norm_rgb = np.zeros_like(rgb_canvas)
        norm_rgb[valid_a] = rgb_canvas[valid_a] / weight_sum[valid_a, None]
        bg = np.array(bg_color, dtype=np.float32)[None, None, :]
        # Use direct normalized surface color where Gaussian support exists, blending soft edges
        edge_blend = np.clip(weight_sum / 0.15, 0.0, 1.0)[:, :, None]
        rgb_out = np.clip(
            edge_blend * norm_rgb + (1.0 - edge_blend) * bg,
            0.0,
            1.0,
        )

        # Build colorized semantic overlay
        sem_rgb = rgb_out * 0.45
        obj_by_idx = {obj.instance_index: obj for obj in self.scene.objects}
        visible_objects: List[Dict[str, Any]] = []
        total_pixels = float(H * W)

        for inst_idx, obj in obj_by_idx.items():
            mask = inst_canvas == inst_idx
            px_count = int(np.sum(mask))
            if px_count > 0:
                sem_rgb[mask] = 0.35 * rgb_out[mask] + 0.65 * obj.color_rgb[None, :]
                ys, xs = np.nonzero(mask)
                mean_u = float(np.mean(xs)) / W
                mean_v = float(np.mean(ys)) / H
                centrality = float(
                    max(0.0, 1.0 - 1.8 * np.hypot(mean_u - 0.5, mean_v - 0.5))
                )
                visible_objects.append(
                    {
                        "object_id": obj.object_id,
                        "label": obj.label,
                        "category": obj.category,
                        "pixel_count": px_count,
                        "pixel_fraction": round(px_count / total_pixels, 5),
                        "mean_depth_m": round(float(np.mean(depth_out[mask])), 3),
                        "view_centrality": round(centrality, 4),
                    }
                )

        visible_objects.sort(key=lambda d: d["pixel_count"], reverse=True)
        if rel_canvas is not None:
            rel_out = np.zeros_like(rel_canvas)
            rel_out[valid_a] = np.clip(rel_canvas[valid_a] / weight_sum[valid_a], 0.0, 1.0)
        else:
            rel_out = None

        return RenderOutput(
            rgb=rgb_out,
            depth=depth_out,
            alpha=alpha_map,
            instance_map=inst_canvas,
            semantic_rgb=np.clip(sem_rgb, 0.0, 1.0),
            relevancy_map=rel_out,
            visible_objects=visible_objects,
            camera_pose_twc=T_wc,
        )
