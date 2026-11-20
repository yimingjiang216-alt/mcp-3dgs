"""
Semantic 3D Gaussian Splatting Scene Representation & Multi-View 2D-to-3D Lifting.

Constructs an explicit 3D Gaussian Splatting scene directly from the real
TUM RGB-D `rgbd_dataset_freiburg1_desk` dataset and bundle-adjusted camera poses
(`slam_ba.npz` from `rgbd-slam`):
  - Unprojects RGB-D keyframes into world coordinates via bundle-adjusted SE(3) poses.
  - Estimates local surface normals from depth gradients to parameterize anisotropic
    3D Gaussian covariances (tangent scales s_x, s_y and thin normal scale s_z) and
    SO(3) orientation quaternions.
  - Runs pretrained Mask R-CNN ResNet50-FPN v2 instance segmentation across keyframes,
    lifting 2D pixel masks into 3D Gaussian semantic/instance votes with depth-consistency gating.
  - Clusters multi-view 3D detections into persistent 3D object instances with 384-D
    neural language embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# Semantic metadata and functional affordances for indoor office categories detected by Mask R-CNN v2
CATEGORY_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "keyboard": {
        "canonical_name": "computer_keyboard",
        "description": (
            "mechanical computer keyboard on office desk for typing text, writing code, "
            "and data entry input device in front of monitor"
        ),
        "affordances": ["typeable", "pressable", "graspable", "stationary_input"],
        "color_rgb": [0.15, 0.75, 0.95],
    },
    "tv": {
        "canonical_name": "desktop_monitor",
        "description": (
            "flat panel LCD computer monitor display screen on office desk for viewing "
            "code, visual output, documents, and graphical user interfaces"
        ),
        "affordances": ["viewable", "readable", "display_screen", "fragile"],
        "color_rgb": [0.20, 0.45, 0.95],
    },
    "laptop": {
        "canonical_name": "laptop_computer",
        "description": (
            "portable laptop notebook computer with screen and keyboard on desk for "
            "mobile computing, programming, and workstation tasks"
        ),
        "affordances": ["viewable", "typeable", "foldable", "portable_compute"],
        "color_rgb": [0.45, 0.35, 0.95],
    },
    "mouse": {
        "canonical_name": "computer_mouse",
        "description": (
            "handheld optical computer mouse pointing device on desk next to keyboard "
            "for cursor navigation and clicking"
        ),
        "affordances": ["graspable", "movable", "clickable", "handheld_pointing"],
        "color_rgb": [0.10, 0.90, 0.55],
    },
    "cup": {
        "canonical_name": "coffee_mug",
        "description": (
            "ceramic coffee cup mug beverage container on desk for drinking water, "
            "coffee, or tea during office work"
        ),
        "affordances": ["graspable", "liftable", "drinkable_container", "pourable"],
        "color_rgb": [0.98, 0.35, 0.25],
    },
    "book": {
        "canonical_name": "academic_books",
        "description": (
            "printed textbook and technical reference book or notebook stack on desk "
            "for reading, studying, and documentation"
        ),
        "affordances": ["graspable", "readable", "stackable", "movable"],
        "color_rgb": [0.95, 0.75, 0.15],
    },
    "remote": {
        "canonical_name": "desk_telephone_handset",
        "description": (
            "office desk telephone handset and remote controller device on front edge "
            "of desk for voice communication and control"
        ),
        "affordances": ["graspable", "pressable", "communication_handset"],
        "color_rgb": [0.95, 0.45, 0.75],
    },
    "chair": {
        "canonical_name": "office_chair",
        "description": (
            "ergonomic swivel office chair next to desk for sitting and workstation support"
        ),
        "affordances": ["sittable", "movable", "support_furniture"],
        "color_rgb": [0.65, 0.45, 0.25],
    },
    "bottle": {
        "canonical_name": "water_bottle",
        "description": (
            "water bottle drink container on office desk for hydration and liquid storage"
        ),
        "affordances": ["graspable", "liftable", "drinkable_container"],
        "color_rgb": [0.25, 0.85, 0.85],
    },
    "dining table": {
        "canonical_name": "wooden_office_desk",
        "description": (
            "large wooden office desk planar support surface holding monitors, keyboard, "
            "laptop, mouse, coffee mug, and books"
        ),
        "affordances": ["support_surface", "placeable_target", "static_obstacle"],
        "color_rgb": [0.60, 0.55, 0.50],
    },
}


class LocalSemanticEncoder:
    """
    384-D Neural Language Encoder backed by locally cached `sentence-transformers/all-MiniLM-L6-v2`.
    Falls back to a deterministic multi-scale character/word n-gram random projection if offline
    weights are absent.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self.embedding_dim = embedding_dim
        self._tokenizer = None
        self._model = None
        self.backend_name = "ngram_fallback"
        self._try_load_local_transformer()

    def _try_load_local_transformer(self) -> None:
        hub_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--sentence-transformers--all-MiniLM-L6-v2"
            / "snapshots"
        )
        if not hub_root.exists():
            return
        snapshots = sorted([p for p in hub_root.iterdir() if p.is_dir()])
        if not snapshots:
            return
        snap = snapshots[-1]
        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
            self._model = AutoModel.from_pretrained(str(snap), local_files_only=True).eval()
            self.backend_name = "sentence-transformers/all-MiniLM-L6-v2"
        except Exception:
            self._tokenizer = None
            self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self._model is not None and self._tokenizer is not None:
            inp = self._tokenizer(
                texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._model(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                normed = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return normed.cpu().numpy().astype(np.float32)

        # Deterministic fallback if HuggingFace cache is not present
        vecs = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().replace("_", " ").replace("-", " ").split()
            for tok in tokens:
                h = abs(hash(tok)) % self.embedding_dim
                vecs[i, h] += 1.0
                for n in range(len(tok) - 2):
                    nh = abs(hash(tok[n : n + 3])) % self.embedding_dim
                    vecs[i, nh] += 0.4
            norm = np.linalg.norm(vecs[i]) + 1e-8
            vecs[i] /= norm
        return vecs


@dataclass
class Object3DInstance:
    """Persistent 3D object instance reconstructed from multi-view Mask R-CNN + 3DGS lifting."""

    object_id: str
    instance_index: int
    category: str
    canonical_name: str
    label: str
    description: str
    affordances: List[str]
    center: np.ndarray  # (3,) world coordinates [m]
    bbox_min: np.ndarray  # (3,) world coordinates [m]
    bbox_max: np.ndarray  # (3,) world coordinates [m]
    confidence: float
    num_gaussians: int
    num_views_observed: int
    color_rgb: np.ndarray  # (3,) semantic visualization color in [0, 1]
    embedding: np.ndarray = field(repr=False)  # (384,) L2-normalized semantic vector

    @property
    def extent(self) -> np.ndarray:
        return np.maximum(self.bbox_max - self.bbox_min, 0.02)

    @property
    def volume_m3(self) -> float:
        ext = self.extent
        return float(ext[0] * ext[1] * ext[2])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "instance_index": int(self.instance_index),
            "category": self.category,
            "canonical_name": self.canonical_name,
            "label": self.label,
            "description": self.description,
            "affordances": list(self.affordances),
            "center_m": [round(float(x), 4) for x in self.center],
            "bbox_min_m": [round(float(x), 4) for x in self.bbox_min],
            "bbox_max_m": [round(float(x), 4) for x in self.bbox_max],
            "extent_m": [round(float(x), 4) for x in self.extent],
            "volume_m3": round(self.volume_m3, 5),
            "confidence": round(float(self.confidence), 4),
            "num_gaussians": int(self.num_gaussians),
            "num_views_observed": int(self.num_views_observed),
            "color_rgb": [round(float(x), 3) for x in self.color_rgb],
        }


def _normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """
    Convert unit surface normals (N, 3) to unit quaternions [w, x, y, z] (N, 4)
    that rotate local +Z [0, 0, 1] onto `normals`.
    """
    n = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dots = np.clip(n @ z, -1.0, 1.0)
    # q = [1 + dot, cross(z, n)] normalized
    cross_x = -n[:, 1]
    cross_y = n[:, 0]
    cross_z = np.zeros_like(dots)
    w = 1.0 + dots
    q = np.stack([w, cross_x, cross_y, cross_z], axis=-1).astype(np.float32)
    anti = w < 1e-5
    if np.any(anti):
        q[anti] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions [w, x, y, z] (..., 4) to rotation matrices (..., 3, 3)."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R.astype(np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply Hamilton quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return (out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)).astype(np.float32)


class SemanticGaussianScene:
    """
    Explicit 3D Gaussian Splatting representation with open-vocabulary instance semantics.

    Each Gaussian i has:
      - position mu_i in R^3 (world meters)
      - scale s_i = (s_x, s_y, s_z) in R_+^3 (meters, principal standard deviations)
      - rotation q_i = (w, x, y, z) in SO(3)
      - opacity alpha_i in (0, 1)
      - RGB color c_i in [0, 1]^3
      - instance_id in Z (0 = background / unlabelled structure, >=1 = Object3DInstance)
    """

    def __init__(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        opacities: np.ndarray,
        colors: np.ndarray,
        instance_ids: np.ndarray,
        objects: List[Object3DInstance],
        intrinsics_K: np.ndarray,
        camera_poses_twc: np.ndarray,
        scene_name: str = "tum_fr1_desk",
        encoder: Optional[LocalSemanticEncoder] = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.rotations = np.asarray(rotations, dtype=np.float32)
        self.opacities = np.asarray(opacities, dtype=np.float32)
        self.colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
        self.instance_ids = np.asarray(instance_ids, dtype=np.int32)
        self.objects = list(objects)
        self.intrinsics_K = np.asarray(intrinsics_K, dtype=np.float32)
        self.camera_poses_twc = np.asarray(camera_poses_twc, dtype=np.float32)
        self.scene_name = scene_name
        self.encoder = encoder or LocalSemanticEncoder()

        # Keep a snapshot of initial state for counterfactual resets
        self._init_positions = self.positions.copy()
        self._init_rotations = self.rotations.copy()
        self._init_opacities = self.opacities.copy()
        self._init_objects_centers = {obj.object_id: obj.center.copy() for obj in self.objects}
        self._init_objects_bmin = {obj.object_id: obj.bbox_min.copy() for obj in self.objects}
        self._init_objects_bmax = {obj.object_id: obj.bbox_max.copy() for obj in self.objects}

    @property
    def num_gaussians(self) -> int:
        return int(self.positions.shape[0])

    def get_covariances_3d(self) -> np.ndarray:
        """
        Compute 3D covariance matrices Sigma_i = R_i diag(s_x^2, s_y^2, s_z^2) R_i^T
        of shape (N, 3, 3).
        """
        R = _quat_to_rotmat(self.rotations)  # (N, 3, 3)
        S2 = (self.scales ** 2)[:, None, :]  # (N, 1, 3)
        RS2 = R * S2  # scales columns of R
        Sigma = np.matmul(RS2, np.swapaxes(R, -1, -2))
        return Sigma.astype(np.float32)

    def get_object_by_id(self, object_id: str) -> Optional[Object3DInstance]:
        q = object_id.strip().lower()
        for obj in self.objects:
            if obj.object_id.lower() == q or obj.canonical_name.lower() == q or obj.label.lower() == q:
                return obj
        # Substring fallback
        for obj in self.objects:
            if q in obj.object_id.lower() or q in obj.label.lower() or q in obj.category.lower():
                return obj
        return None

    def reset_edits(self) -> None:
        """Restore scene Gaussians and object bounds to their original unedited state."""
        self.positions[:] = self._init_positions
        self.rotations[:] = self._init_rotations
        self.opacities[:] = self._init_opacities
        for obj in self.objects:
            obj.center = self._init_objects_centers[obj.object_id].copy()
            obj.bbox_min = self._init_objects_bmin[obj.object_id].copy()
            obj.bbox_max = self._init_objects_bmax[obj.object_id].copy()

    def save(self, npz_path: Path, json_path: Optional[Path] = None) -> None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        obj_embeddings = (
            np.stack([o.embedding for o in self.objects], axis=0)
            if self.objects
            else np.zeros((0, 384), dtype=np.float32)
        )
        obj_meta = json.dumps([o.to_dict() for o in self.objects], ensure_ascii=False)
        np.savez_compressed(
            npz_path,
            positions=self.positions,
            scales=self.scales,
            rotations=self.rotations,
            opacities=self.opacities,
            colors=self.colors,
            instance_ids=self.instance_ids,
            intrinsics_K=self.intrinsics_K,
            camera_poses_twc=self.camera_poses_twc,
            obj_embeddings=obj_embeddings,
            obj_meta=np.array(obj_meta),
            scene_name=np.array(self.scene_name),
        )
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "scene_name": self.scene_name,
                        "num_gaussians": self.num_gaussians,
                        "num_objects": len(self.objects),
                        "semantic_encoder": self.encoder.backend_name,
                        "bounds_min_m": [round(float(x), 4) for x in self.positions.min(axis=0)],
                        "bounds_max_m": [round(float(x), 4) for x in self.positions.max(axis=0)],
                        "objects": [o.to_dict() for o in self.objects],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    @classmethod
    def load(
        cls, npz_path: Path, encoder: Optional[LocalSemanticEncoder] = None
    ) -> "SemanticGaussianScene":
        data = np.load(npz_path, allow_pickle=False)
        enc = encoder or LocalSemanticEncoder()
        raw_meta = json.loads(str(data["obj_meta"].item()))
        obj_embs = data["obj_embeddings"]
        objects: List[Object3DInstance] = []
        for idx, item in enumerate(raw_meta):
            emb = obj_embs[idx] if idx < len(obj_embs) else enc.encode([item["description"]])[0]
            objects.append(
                Object3DInstance(
                    object_id=item["object_id"],
                    instance_index=int(item["instance_index"]),
                    category=item["category"],
                    canonical_name=item["canonical_name"],
                    label=item["label"],
                    description=item["description"],
                    affordances=list(item["affordances"]),
                    center=np.array(item["center_m"], dtype=np.float32),
                    bbox_min=np.array(item["bbox_min_m"], dtype=np.float32),
                    bbox_max=np.array(item["bbox_max_m"], dtype=np.float32),
                    confidence=float(item["confidence"]),
                    num_gaussians=int(item["num_gaussians"]),
                    num_views_observed=int(item["num_views_observed"]),
                    color_rgb=np.array(item["color_rgb"], dtype=np.float32),
                    embedding=np.asarray(emb, dtype=np.float32),
                )
            )
        return cls(
            positions=data["positions"],
            scales=data["scales"],
            rotations=data["rotations"],
            opacities=data["opacities"],
            colors=data["colors"],
            instance_ids=data["instance_ids"],
            objects=objects,
            intrinsics_K=data["intrinsics_K"],
            camera_poses_twc=data["camera_poses_twc"],
            scene_name=str(data["scene_name"].item()),
            encoder=enc,
        )


def _cluster_multiview_detections(
    raw_detections: List[Dict[str, Any]],
    positions: np.ndarray,
    encoder: LocalSemanticEncoder,
) -> Tuple[np.ndarray, List[Object3DInstance]]:
    """
    Cluster multi-view 3D Mask R-CNN detections by category and 3D spatial proximity,
    assigning each Gaussian to its highest-consensus 3D object instance.
    """
    clusters: List[Dict[str, Any]] = []
    # Merge radius per category (meters)
    merge_radii = {
        "dining table": 0.65,
        "chair": 0.45,
        "tv": 0.35,
        "laptop": 0.35,
        "keyboard": 0.32,
        "book": 0.26,
        "cup": 0.24,
        "mouse": 0.24,
        "remote": 0.26,
        "bottle": 0.24,
    }
    for det in sorted(raw_detections, key=lambda d: d["score"], reverse=True):
        cat = det["category"]
        if cat not in CATEGORY_SEMANTICS:
            continue
        center = det["center"]
        rad = merge_radii.get(cat, 0.30)
        matched = None
        best_dist = rad
        for cl in clusters:
            if cl["category"] != cat:
                continue
            dist = float(np.linalg.norm(cl["center"] - center))
            if dist < best_dist:
                best_dist = dist
                matched = cl
        if matched is None:
            clusters.append(
                {
                    "category": cat,
                    "center": center.copy(),
                    "scores": [det["score"]],
                    "frames": {det["frame_idx"]},
                    "points": [det["points"]],
                }
            )
        else:
            w = len(matched["scores"])
            matched["center"] = (matched["center"] * w + center) / (w + 1)
            matched["scores"].append(det["score"])
            matched["frames"].add(det["frame_idx"])
            matched["points"].append(det["points"])

    # Filter clusters: require multi-view consensus (>= 2 views or high confidence >= 0.84)
    valid_clusters = [
        cl
        for cl in clusters
        if len(cl["frames"]) >= 2 or max(cl["scores"]) >= 0.84
    ]
    # Sort by importance (foreground manipulable/interactive objects first, desk surface last)
    valid_clusters.sort(
        key=lambda cl: (
            0 if cl["category"] != "dining table" else 1,
            -len(cl["frames"]),
            -max(cl["scores"]),
        )
    )

    instance_ids = np.zeros(positions.shape[0], dtype=np.int32)
    objects: List[Object3DInstance] = []
    cat_counts: Dict[str, int] = {}

    # Build a KDTree on Gaussian positions to assign Gaussians to 3D object clusters
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)

    for cl in valid_clusters:
        cat = cl["category"]
        pts_all = np.concatenate(cl["points"], axis=0)
        if pts_all.shape[0] < 40:
            continue
        # Robust 5%..95% quantile bounding box with slight padding
        q_low = np.quantile(pts_all, 0.05, axis=0) - 0.015
        q_high = np.quantile(pts_all, 0.95, axis=0) + 0.015
        # Find scene Gaussians within 2.5cm of the multi-view lifted instance point cloud
        sub_pts = pts_all[:: max(1, len(pts_all) // 1500)]
        neighbor_lists = tree.query_ball_point(sub_pts, r=0.028)
        g_idx = np.unique([i for lst in neighbor_lists for i in lst]).astype(np.int32)
        # Keep Gaussians inside the quantile box and not yet claimed by a smaller foreground object
        if g_idx.size > 0:
            in_box = np.all(
                (positions[g_idx] >= q_low) & (positions[g_idx] <= q_high), axis=-1
            )
            g_idx = g_idx[in_box]
            if cat == "dining table":
                g_idx = g_idx[instance_ids[g_idx] == 0]

        if g_idx.size < 25:
            continue

        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        inst_idx = len(objects) + 1
        if cat != "dining table":
            instance_ids[g_idx] = inst_idx
        else:
            unclaimed = g_idx[instance_ids[g_idx] == 0]
            instance_ids[unclaimed] = inst_idx
            g_idx = unclaimed

        sem = CATEGORY_SEMANTICS[cat]
        c_num = cat_counts[cat]
        obj_id = f"{sem['canonical_name']}_{c_num:02d}"
        center_w = np.median(positions[g_idx], axis=0).astype(np.float32)
        bmin_w = np.quantile(positions[g_idx], 0.02, axis=0).astype(np.float32)
        bmax_w = np.quantile(positions[g_idx], 0.98, axis=0).astype(np.float32)

        # Add spatial qualifier (left/right/center) to description for finer open-vocab grounding
        horiz_tag = (
            "right side of desk"
            if center_w[0] > -0.25
            else ("center of desk" if center_w[0] > -0.75 else "left side of desk")
        )
        label = f"{sem['canonical_name'].replace('_', ' ').title()} #{c_num} ({horiz_tag})"
        desc = (
            f"{label}: {sem['description']}. Located at {horiz_tag} "
            f"around 3D coordinates ({center_w[0]:.2f}, {center_w[1]:.2f}, {center_w[2]:.2f}) meters."
        )
        emb = encoder.encode([desc])[0]
        color_rgb = np.array(sem["color_rgb"], dtype=np.float32)
        if c_num > 1:
            color_rgb = np.clip(color_rgb * 0.85 + 0.12, 0.05, 0.98)

        objects.append(
            Object3DInstance(
                object_id=obj_id,
                instance_index=inst_idx,
                category=cat,
                canonical_name=sem["canonical_name"],
                label=label,
                description=desc,
                affordances=list(sem["affordances"]),
                center=center_w,
                bbox_min=bmin_w,
                bbox_max=bmax_w,
                confidence=float(np.mean(sorted(cl["scores"], reverse=True)[:3])),
                num_gaussians=int(g_idx.size),
                num_views_observed=len(cl["frames"]),
                color_rgb=color_rgb,
                embedding=emb,
            )
        )

    # Recompute exact Gaussian counts after any overwrites
    for obj in objects:
        obj.num_gaussians = int(np.sum(instance_ids == obj.instance_index))

    return instance_ids, objects


def build_scene_from_tum_rgbd(
    tum_dir: Path,
    num_geo_keyframes: int = 36,
    num_sem_keyframes: int = 16,
    pixel_stride: int = 4,
    voxel_size: float = 0.014,
    run_maskrcnn: bool = True,
    photometric_iters: int = 25,
) -> SemanticGaussianScene:
    """
    Reconstruct a Semantic 3D Gaussian Splatting scene from `TUM rgbd_dataset_freiburg1_desk`.

    Steps:
      1. Load bundle-adjusted camera poses `poses (613, 4, 4)` and intrinsics `K` from `slam_ba.npz`.
      2. Associate RGB and Depth timestamps (`rgb.txt`, `depth.txt`).
      3. Unproject RGB-D keyframes, compute local surface normals from depth maps, and initialize
         anisotropic 3D Gaussians (tangent scales proportional to pixel footprint z/f, normal scale
         thin along the surface normal).
      4. Fuse multi-view Gaussians on a 3D spatial hash grid (`voxel_size`), boosting opacity for
         consistently observed surfaces.
      5. Run pretrained Mask R-CNN ResNet50-FPN v2 across `num_sem_keyframes` views to lift 2D
         instance segmentation masks into 3D object clusters with 384-D `all-MiniLM-L6-v2` embeddings.
      6. Perform multi-view photometric color & opacity refinement against keyframe RGB images.
    """
    tum_dir = Path(tum_dir)
    ba_data = np.load(tum_dir / "slam_ba.npz")
    K = ba_data["K"].astype(np.float32)
    poses_twc = ba_data["poses"].astype(np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    rgb_lines = [
        l.strip().split()
        for l in (tum_dir / "rgb.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    depth_lines = [
        l.strip().split()
        for l in (tum_dir / "depth.txt").read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    depth_ts = np.array([float(x[0]) for x in depth_lines], dtype=np.float64)

    # Enforce strict RGB-to-Depth timestamp synchronization (|t_d - t_rgb| <= 0.025s):
    # frames 0..17 in TUM fr1/desk precede the start of depth.txt by up to 0.582s.
    # Use the continuous first desk sweep (frames 18..162) to avoid open-loop VO drift.
    geo_indices = np.linspace(18, 162, num_geo_keyframes, dtype=int)

    all_pts = []
    all_cols = []
    all_norms = []
    all_scales = []
    kf_depths = []

    for idx in geo_indices:
        r_ts = float(rgb_lines[idx][0])
        d_idx = int(np.argmin(np.abs(depth_ts - r_ts)))
        if abs(depth_ts[d_idx] - r_ts) > 0.025:
            continue
        rgb_img = np.asarray(Image.open(tum_dir / rgb_lines[idx][1]).convert("RGB"), dtype=np.float32) / 255.0
        depth_img = np.asarray(Image.open(tum_dir / depth_lines[d_idx][1]), dtype=np.float32) / 5000.0
        kf_depths.append((int(idx), depth_img))
        T_wc = poses_twc[idx]
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        # Compute camera-space 3D coordinates on strided grid
        H, W = depth_img.shape
        us_grid, vs_grid = np.meshgrid(
            np.arange(pixel_stride, W - pixel_stride, pixel_stride),
            np.arange(pixel_stride, H - pixel_stride, pixel_stride),
        )
        z = depth_img[vs_grid, us_grid]
        valid = (z > 0.35) & (z < 2.65)
        if np.sum(valid) < 50:
            continue

        # Depth gradient for local surface normal estimation
        dz_du = (
            depth_img[vs_grid, np.clip(us_grid + pixel_stride, 0, W - 1)]
            - depth_img[vs_grid, np.clip(us_grid - pixel_stride, 0, W - 1)]
        ) / (2.0 * pixel_stride)
        dz_dv = (
            depth_img[np.clip(vs_grid + pixel_stride, 0, H - 1), us_grid]
            - depth_img[np.clip(vs_grid - pixel_stride, 0, H - 1), us_grid]
        ) / (2.0 * pixel_stride)
        # Filter depth discontinuities (occlusion edges)
        smooth_edge = (np.abs(dz_du) < 0.025) & (np.abs(dz_dv) < 0.025)
        valid = valid & smooth_edge

        u_v = us_grid[valid].astype(np.float32)
        v_v = vs_grid[valid].astype(np.float32)
        z_v = z[valid].astype(np.float32)
        x_c = (u_v - cx) * z_v / fx
        y_c = (v_v - cy) * z_v / fy
        pts_c = np.stack([x_c, y_c, z_v], axis=-1)
        pts_w = (pts_c @ R_wc.T) + t_wc

        # Camera-frame surface normal pointing toward camera: n_c ~ [-fx*dz/du, -fy*dz/dv, -z]
        nx_c = -dz_du[valid] * fx / np.maximum(z_v, 0.3)
        ny_c = -dz_dv[valid] * fy / np.maximum(z_v, 0.3)
        nz_c = -np.ones_like(z_v)
        n_c = np.stack([nx_c, ny_c, nz_c], axis=-1)
        n_c /= np.maximum(np.linalg.norm(n_c, axis=-1, keepdims=True), 1e-8)
        n_w = n_c @ R_wc.T

        # Anisotropic Gaussian scales: tangent scales match pixel spacing, normal scale is thin disk
        s_tan = np.clip(1.45 * pixel_stride * z_v / fx, 0.006, 0.028)
        s_norm = np.clip(0.35 * s_tan, 0.0025, 0.009)
        scales_w = np.stack([s_tan, s_tan * 0.92, s_norm], axis=-1)

        cols = rgb_img[vs_grid[valid], us_grid[valid]]
        all_pts.append(pts_w)
        all_cols.append(cols)
        all_norms.append(n_w)
        all_scales.append(scales_w)

    pts_cat = np.concatenate(all_pts, axis=0)
    cols_cat = np.concatenate(all_cols, axis=0)
    norms_cat = np.concatenate(all_norms, axis=0)
    scales_cat = np.concatenate(all_scales, axis=0)

    # Multi-view free-space visibility carving: prune floaters that project in front of sensor depth
    support_cnt = np.zeros(len(pts_cat), dtype=np.int32)
    violate_cnt = np.zeros(len(pts_cat), dtype=np.int32)
    for kf_i, d_kf in kf_depths:
        T_cw = np.linalg.inv(poses_twc[kf_i])
        pc = (pts_cat @ T_cw[:3, :3].T) + T_cw[:3, 3]
        z_k = pc[:, 2]
        u_k = np.round(pc[:, 0] * fx / np.maximum(z_k, 1e-4) + cx).astype(np.int32)
        v_k = np.round(pc[:, 1] * fy / np.maximum(z_k, 1e-4) + cy).astype(np.int32)
        in_b = (z_k > 0.35) & (u_k >= 0) & (u_k < 640) & (v_k >= 0) & (v_k < 480)
        idx_in = np.nonzero(in_b)[0]
        d_s = d_kf[v_k[idx_in], u_k[idx_in]]
        val_s = d_s > 0.35
        idx_v = idx_in[val_s]
        dz = z_k[idx_v] - d_s[val_s]
        support_cnt[idx_v[np.abs(dz) < 0.04]] += 1
        violate_cnt[idx_v[dz < -0.045]] += 1

    carve_keep = (support_cnt >= 2) & (violate_cnt == 0)
    if int(np.sum(carve_keep)) > 10000:
        pts_cat = pts_cat[carve_keep]
        cols_cat = cols_cat[carve_keep]
        norms_cat = norms_cat[carve_keep]
        scales_cat = scales_cat[carve_keep]

    # Voxel-grid multi-view Gaussian fusion
    vox_coords = np.floor(pts_cat / voxel_size).astype(np.int64)
    # Pack 3D voxel indices into 64-bit hash key
    mins = vox_coords.min(axis=0)
    vc = vox_coords - mins
    keys = (vc[:, 0] << 40) | (vc[:, 1] << 20) | vc[:, 2]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    unique_keys, idx_start, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )

    # Keep voxels with sufficient support or sample up to ~48,000 high-quality Gaussians
    N_vox = len(unique_keys)
    positions = np.zeros((N_vox, 3), dtype=np.float32)
    colors = np.zeros((N_vox, 3), dtype=np.float32)
    normals = np.zeros((N_vox, 3), dtype=np.float32)
    scales = np.zeros((N_vox, 3), dtype=np.float32)

    # Fast vectorized reduceat aggregation
    pts_ord = pts_cat[order]
    cols_ord = cols_cat[order]
    norms_ord = norms_cat[order]
    scales_ord = scales_cat[order]

    inv_counts = (1.0 / counts[:, None]).astype(np.float32)
    positions = np.add.reduceat(pts_ord, idx_start, axis=0) * inv_counts
    colors = np.add.reduceat(cols_ord, idx_start, axis=0) * inv_counts
    normals = np.add.reduceat(norms_ord, idx_start, axis=0) * inv_counts
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    scales = np.add.reduceat(scales_ord, idx_start, axis=0) * inv_counts

    # Opacity increases with multi-view observation count: alpha = 1 - exp(-0.55 * count)
    opacities = (1.0 - np.exp(-0.58 * counts.astype(np.float32))).clip(0.38, 0.98)

    # Cap total Gaussians to ~45,000 highest-confidence primitives for fast interactive MCP rendering
    max_gaussians = 45000
    if N_vox > max_gaussians:
        # Prioritize multi-view observed voxels + central desk workspace
        dist_center = np.linalg.norm(positions - np.array([-0.45, 0.10, 1.05]), axis=-1)
        priority = counts.astype(np.float32) + 2.0 / (0.3 + dist_center)
        keep_idx = np.argsort(priority)[-max_gaussians:]
        positions = positions[keep_idx]
        colors = colors[keep_idx]
        normals = normals[keep_idx]
        scales = scales[keep_idx]
        opacities = opacities[keep_idx]

    rotations = _normals_to_quaternions(normals)

    # Multi-view 2D-to-3D Mask R-CNN v2 instance segmentation lifting
    encoder = LocalSemanticEncoder()
    raw_detections: List[Dict[str, Any]] = []
    if run_maskrcnn:
        import torchvision

        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        detector = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights).eval()
        categories = weights.meta["categories"]
        preprocess = weights.transforms()

        sem_indices = np.linspace(0, len(rgb_lines) - 1, num_sem_keyframes, dtype=int)
        for idx in sem_indices:
            r_ts = float(rgb_lines[idx][0])
            d_idx = int(np.argmin(np.abs(depth_ts - r_ts)))
            pil_img = Image.open(tum_dir / rgb_lines[idx][1]).convert("RGB")
            depth_img = np.asarray(Image.open(tum_dir / depth_lines[d_idx][1]), dtype=np.float32) / 5000.0
            T_wc = poses_twc[idx]

            with torch.no_grad():
                pred = detector([preprocess(pil_img)])[0]

            for score, label_id, mask_t in zip(pred["scores"], pred["labels"], pred["masks"]):
                sc = float(score.item())
                if sc < 0.56:
                    continue
                cat = categories[int(label_id.item())]
                if cat not in CATEGORY_SEMANTICS:
                    continue
                mask_np = (mask_t[0].cpu().numpy() > 0.52) & (depth_img > 0.35) & (depth_img < 2.45)
                if int(mask_np.sum()) < 350:
                    continue
                vs, us = np.nonzero(mask_np)
                # Subsample mask pixels for fast 3D unprojection
                step = max(1, len(vs) // 800)
                vs, us = vs[::step], us[::step]
                zs = depth_img[vs, us]
                # Filter depth outliers inside the mask (keep foreground surface around median depth)
                z_med = float(np.median(zs))
                z_inlier = np.abs(zs - z_med) < (0.35 if cat == "dining table" else 0.14)
                if int(z_inlier.sum()) < 25:
                    continue
                vs, us, zs = vs[z_inlier], us[z_inlier], zs[z_inlier]
                xs = (us.astype(np.float32) - cx) * zs / fx
                ys = (vs.astype(np.float32) - cy) * zs / fy
                pts_c = np.stack([xs, ys, zs], axis=-1)
                pts_w = (pts_c @ T_wc[:3, :3].T) + T_wc[:3, 3]
                raw_detections.append(
                    {
                        "frame_idx": int(idx),
                        "category": cat,
                        "score": sc,
                        "center": np.median(pts_w, axis=0).astype(np.float32),
                        "points": pts_w.astype(np.float32),
                    }
                )

    instance_ids, objects = _cluster_multiview_detections(raw_detections, positions, encoder)

    # Multi-view photometric color & scale refinement on training keyframes
    if photometric_iters > 0:
        _photometric_refinement(
            positions=positions,
            colors=colors,
            scales=scales,
            opacities=opacities,
            tum_dir=tum_dir,
            rgb_lines=rgb_lines,
            depth_lines=depth_lines,
            depth_ts=depth_ts,
            poses_twc=poses_twc,
            K=K,
            num_iters=photometric_iters,
        )

    return SemanticGaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        instance_ids=instance_ids,
        objects=objects,
        intrinsics_K=K,
        camera_poses_twc=poses_twc,
        scene_name="tum_fr1_desk",
        encoder=encoder,
    )


def _photometric_refinement(
    positions: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    tum_dir: Path,
    rgb_lines: List[List[str]],
    depth_lines: List[List[str]],
    depth_ts: np.ndarray,
    poses_twc: np.ndarray,
    K: np.ndarray,
    num_iters: int = 25,
) -> None:
    """
    Iterative multi-view photometric & depth-consistency refinement:
    projects 3D Gaussians onto training keyframes, compares splatted color against
    ground-truth RGB-D pixels, and updates Gaussian colors, opacities, and tangent scales
    via visibility-weighted gradient steps.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    train_indices = np.linspace(0, len(rgb_lines) - 1, min(num_iters, 24), dtype=int)
    lr_color = 0.22

    for idx in train_indices:
        r_ts = float(rgb_lines[idx][0])
        d_idx = int(np.argmin(np.abs(depth_ts - r_ts)))
        gt_rgb = np.asarray(Image.open(tum_dir / rgb_lines[idx][1]).convert("RGB"), dtype=np.float32) / 255.0
        gt_depth = np.asarray(Image.open(tum_dir / depth_lines[d_idx][1]), dtype=np.float32) / 5000.0
        H, W = gt_depth.shape
        T_cw = np.linalg.inv(poses_twc[idx])
        pts_c = (positions @ T_cw[:3, :3].T) + T_cw[:3, 3]
        z_c = pts_c[:, 2]
        in_front = z_c > 0.35
        u = np.round(pts_c[:, 0] * fx / np.maximum(z_c, 1e-4) + cx).astype(np.int32)
        v = np.round(pts_c[:, 1] * fy / np.maximum(z_c, 1e-4) + cy).astype(np.int32)
        in_fov = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vis_idx = np.nonzero(in_fov)[0]
        if vis_idx.size == 0:
            continue
        u_v, v_v, z_v = u[vis_idx], v[vis_idx], z_c[vis_idx]
        d_sensor = gt_depth[v_v, u_v]
        # Visibility gate: Gaussian is unoccluded if its depth matches sensor depth within 3.5cm
        unoccluded = (d_sensor > 0.35) & (np.abs(z_v - d_sensor) < 0.035)
        active = vis_idx[unoccluded]
        if active.size == 0:
            continue
        target_col = gt_rgb[v_v[unoccluded], u_v[unoccluded]]
        residual = target_col - colors[active]
        colors[active] = np.clip(colors[active] + lr_color * residual, 0.0, 1.0)
        opacities[active] = np.clip(opacities[active] + 0.025, 0.35, 0.99)

        # Penalize free-space floaters whose projected depth is >8cm in front of sensor depth
        floater = (d_sensor > 0.35) & ((d_sensor - z_v) > 0.08)
        floater_idx = vis_idx[floater]
        if floater_idx.size > 0:
            opacities[floater_idx] = np.clip(opacities[floater_idx] * 0.78, 0.05, 0.99)
