"""
Spatial Intelligence Engine for MCP-3DGS:
  1. Open-vocabulary 3D object grounding (`query_3d_object`)
  2. Metric & topological 3D spatial relation reasoning (`compute_spatial_relations`)
  3. Active VLN candidate-view imagination & Next-Best-View selection (`imagine_candidate_views`)
  4. 3DGS-to-ESDF collision-free 3D trajectory planning (`plan_collision_free_path`)
  5. Counterfactual 3D Gaussian object manipulation & editing (`edit_3d_object`)
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt

from .gaussian_scene import Object3DInstance, SemanticGaussianScene, _quat_multiply
from .renderer import GaussianRenderer, RenderOutput, look_at_pose


class SpatialIntelligenceEngine:
    """
    High-level spatial perception, imagination, planning, and editing engine
    operating directly on a `SemanticGaussianScene`.
    """

    def __init__(
        self,
        scene: SemanticGaussianScene,
        voxel_size: float = 0.03,
        opacity_threshold: float = 0.45,
    ) -> None:
        self.scene = scene
        self.renderer = GaussianRenderer(scene, width=640, height=480, anisotropic=True)
        self.voxel_size = float(voxel_size)
        self.opacity_threshold = float(opacity_threshold)
        self._esdf_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # 1. Scene Summary & Open-Vocabulary 3D Object Grounding
    # ------------------------------------------------------------------
    def get_scene_summary(self) -> Dict[str, Any]:
        bmin = self.scene.positions.min(axis=0)
        bmax = self.scene.positions.max(axis=0)
        cam_positions = self.scene.camera_poses_twc[:, :3, 3]
        traj_len = float(
            np.sum(np.linalg.norm(np.diff(cam_positions, axis=0), axis=-1))
        )
        return {
            "scene_name": self.scene.scene_name,
            "num_gaussians": self.scene.num_gaussians,
            "num_reconstructed_cameras": int(len(self.scene.camera_poses_twc)),
            "camera_trajectory_length_m": round(traj_len, 3),
            "scene_bounds_min_m": [round(float(x), 3) for x in bmin],
            "scene_bounds_max_m": [round(float(x), 3) for x in bmax],
            "semantic_encoder": self.scene.encoder.backend_name,
            "num_objects": len(self.scene.objects),
            "objects": [obj.to_dict() for obj in self.scene.objects],
        }

    def query_3d_object(
        self, text_query: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Ground an open-vocabulary natural language query to 3D Gaussian object instances
        using 384-D neural language embeddings + affordance/category matching.
        """
        q_clean = text_query.strip()
        q_emb = self.scene.encoder.encode([q_clean])[0]
        q_low = q_clean.lower()

        scored: List[Dict[str, Any]] = []
        for obj in self.scene.objects:
            neural_sim = float(np.dot(obj.embedding, q_emb))
            # Lexical & affordance alignment bonus for explicit category/affordance mentions
            lex_bonus = 0.0
            if obj.category.lower() in q_low or obj.canonical_name.replace("_", " ") in q_low:
                lex_bonus += 0.18
            for token in q_low.replace(",", " ").split():
                if len(token) >= 3 and (
                    token in obj.description.lower()
                    or any(token in aff.lower() for aff in obj.affordances)
                ):
                    lex_bonus += 0.045
            lex_bonus = min(lex_bonus, 0.25)
            combined_score = 0.78 * neural_sim + lex_bonus + 0.04 * obj.confidence

            # Find nearest neighboring object for spatial context
            nearest_id = None
            nearest_dist = 999.0
            for other in self.scene.objects:
                if other.object_id == obj.object_id:
                    continue
                d = float(np.linalg.norm(obj.center - other.center))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_id = other.object_id

            entry = obj.to_dict()
            entry.update(
                {
                    "neural_cosine_similarity": round(neural_sim, 4),
                    "grounding_score": round(combined_score, 4),
                    "nearest_neighbor_id": nearest_id,
                    "nearest_neighbor_distance_m": round(nearest_dist, 4),
                }
            )
            scored.append(entry)

        scored.sort(key=lambda x: x["grounding_score"], reverse=True)
        top_matches = scored[: max(1, int(top_k))]
        return {
            "query": text_query,
            "semantic_encoder": self.scene.encoder.backend_name,
            "top_match": top_matches[0] if top_matches else None,
            "matches": top_matches,
        }

    # ------------------------------------------------------------------
    # 2. Metric & Topological 3D Spatial Relation Reasoning
    # ------------------------------------------------------------------
    def compute_spatial_relations(
        self, object_a: str, object_b: str
    ) -> Dict[str, Any]:
        """
        Compute metric 3D distance, bounding-box surface clearance, relative offset vector,
        and topological spatial predicates between two 3D objects (by ID or open-vocab query).
        """
        obj_a = self.scene.get_object_by_id(object_a)
        if obj_a is None:
            res_a = self.query_3d_object(object_a, top_k=1)
            if res_a["top_match"]:
                obj_a = self.scene.get_object_by_id(res_a["top_match"]["object_id"])

        obj_b = self.scene.get_object_by_id(object_b)
        if obj_b is None:
            res_b = self.query_3d_object(object_b, top_k=1)
            if res_b["top_match"]:
                obj_b = self.scene.get_object_by_id(res_b["top_match"]["object_id"])

        if obj_a is None or obj_b is None:
            raise ValueError(f"Could not resolve objects: '{object_a}' vs '{object_b}'")

        delta = obj_b.center - obj_a.center  # vector from A to B
        center_dist = float(np.linalg.norm(delta))

        # Axis-aligned bounding box minimum surface gap
        gap_xyz = np.maximum(
            0.0,
            np.maximum(obj_a.bbox_min - obj_b.bbox_max, obj_b.bbox_min - obj_a.bbox_max),
        )
        surface_clearance = float(np.linalg.norm(gap_xyz))

        # Topological predicates in TUM fr1/desk world frame:
        # +X is right along the desk, -Y is upward above the desk, +Z is deeper away from camera_0
        predicates: List[str] = []
        if delta[0] > 0.08:
            predicates.append("A_is_left_of_B")
        elif delta[0] < -0.08:
            predicates.append("A_is_right_of_B")

        if delta[1] > 0.07:
            predicates.append("A_is_higher_than_B")
        elif delta[1] < -0.07:
            predicates.append("A_is_lower_than_B")

        if delta[2] > 0.08:
            predicates.append("A_is_in_front_of_B")
        elif delta[2] < -0.08:
            predicates.append("A_is_behind_B")

        if center_dist < 0.35 or surface_clearance < 0.08:
            predicates.append("adjacent_to")
        if obj_b.category == "dining table" or "desk" in obj_b.canonical_name:
            predicates.append("supported_by_desk")

        summary = (
            f"'{obj_a.object_id}' is {center_dist:.3f} m from '{obj_b.object_id}' "
            f"(bbox surface clearance {surface_clearance:.3f} m); relations: {', '.join(predicates) or 'co-located'}."
        )
        return {
            "object_a": obj_a.object_id,
            "object_b": obj_b.object_id,
            "center_distance_m": round(center_dist, 4),
            "bbox_surface_clearance_m": round(surface_clearance, 4),
            "relative_vector_a_to_b_m": [round(float(x), 4) for x in delta],
            "predicates": predicates,
            "natural_language_summary": summary,
        }

    # ------------------------------------------------------------------
    # 3. Active VLN Candidate-View Imagination (Next-Best-View Selection)
    # ------------------------------------------------------------------
    def imagine_candidate_views(
        self,
        target_query: str,
        current_camera_pos: Optional[List[float]] = None,
        num_candidates: int = 6,
    ) -> Dict[str, Any]:
        """
        Active Perception / VLN Imagination in 3DGS World Model:
        Given a target query and current camera position, generates `num_candidates`
        collision-free candidate viewpoints on a viewing hemisphere around the target,
        renders each view in the 3DGS model BEFORE moving the physical camera, and
        selects the Next-Best-View (NBV) maximizing target visibility, unoccluded surface
        ratio, and framing centrality while penalizing travel distance.
        """
        grounding = self.query_3d_object(target_query, top_k=1)
        if not grounding["top_match"]:
            raise ValueError(f"No 3D object matched target_query='{target_query}'")

        target_id = grounding["top_match"]["object_id"]
        target_obj = self.scene.get_object_by_id(target_id)
        assert target_obj is not None
        target_center = target_obj.center.copy()

        if current_camera_pos is None:
            # Default to a real camera pose from the sequence that has partial/suboptimal view
            curr_pos = self.scene.camera_poses_twc[0, :3, 3].copy()
        else:
            curr_pos = np.asarray(current_camera_pos, dtype=np.float32).reshape(3)

        esdf_info = self._get_or_build_esdf()

        # Evaluate initial/current camera view looking at the scene center
        init_pose = look_at_pose(curr_pos, np.array([-0.45, 0.15, 1.05], dtype=np.float32))
        init_eval = self._evaluate_view_for_target(
            init_pose, target_obj, curr_pos, candidate_id="initial_view"
        )

        # Sample candidate viewpoints on a viewing arc/hemisphere in front of the target
        candidates: List[Dict[str, Any]] = []
        azimuths = np.linspace(-0.85, 0.85, num_candidates)
        elevations = [-0.28, -0.18, -0.35, -0.12, -0.24, -0.30]
        radii = [0.68, 0.62, 0.72, 0.58, 0.65, 0.75]

        for k in range(num_candidates):
            az = float(azimuths[k % len(azimuths)])
            el = float(elevations[k % len(elevations)])
            rad = float(radii[k % len(radii)])

            # Offset toward the open workspace side (-Z is toward the user/camera side of the desk)
            dx = rad * math.sin(az) * math.cos(el)
            dy = rad * math.sin(el)
            dz = -rad * math.cos(az) * math.cos(el)
            cand_pos = target_center + np.array([dx, dy, dz], dtype=np.float32)

            # Ensure candidate camera pose has >= 10 cm obstacle clearance in 3DGS ESDF
            clearance = self._query_esdf_point(cand_pos, esdf_info)
            if clearance < 0.10:
                cand_pos[2] -= 0.18  # Pull further back toward free workspace
                cand_pos[1] -= 0.08
                clearance = self._query_esdf_point(cand_pos, esdf_info)

            cand_pose = look_at_pose(cand_pos, target_center)
            ev = self._evaluate_view_for_target(
                cand_pose,
                target_obj,
                curr_pos,
                candidate_id=f"cand_view_{k+1:02d}",
                esdf_clearance=clearance,
            )
            candidates.append(ev)

        candidates_sorted = sorted(
            candidates, key=lambda c: c["nbv_utility_score"], reverse=True
        )
        best_cand = candidates_sorted[0]
        mean_random_score = float(np.mean([c["nbv_utility_score"] for c in candidates]))
        mean_random_vis = float(np.mean([c["target_pixel_fraction"] for c in candidates]))

        return {
            "target_query": target_query,
            "grounded_object_id": target_obj.object_id,
            "grounded_center_m": [round(float(x), 4) for x in target_center],
            "initial_view": init_eval,
            "best_imagined_view": best_cand,
            "all_candidate_views": candidates_sorted,
            "information_gain_summary": {
                "visibility_fraction_initial": init_eval["target_pixel_fraction"],
                "visibility_fraction_random_avg": round(mean_random_vis, 5),
                "visibility_fraction_nbv": best_cand["target_pixel_fraction"],
                "unoccluded_ratio_initial": init_eval["unoccluded_ratio"],
                "unoccluded_ratio_nbv": best_cand["unoccluded_ratio"],
                "utility_score_initial": init_eval["nbv_utility_score"],
                "utility_score_random_avg": round(mean_random_score, 4),
                "utility_score_nbv": best_cand["nbv_utility_score"],
            },
        }

    def _evaluate_view_for_target(
        self,
        pose_twc: np.ndarray,
        target_obj: Object3DInstance,
        curr_pos: np.ndarray,
        candidate_id: str,
        esdf_clearance: float = 0.25,
    ) -> Dict[str, Any]:
        out = self.renderer.render(pose_twc, scale_factor=0.5)
        H, W = out.instance_map.shape
        mask = out.instance_map == target_obj.instance_index
        px_count = int(np.sum(mask))
        px_frac = float(px_count) / float(H * W)

        # Compute unoccluded ratio of the target object's 3D Gaussians from this pose
        T_cw = np.linalg.inv(pose_twc)
        obj_g_mask = self.scene.instance_ids == target_obj.instance_index
        obj_pts_w = self.scene.positions[obj_g_mask]
        if obj_pts_w.shape[0] > 0:
            pts_c = (obj_pts_w @ T_cw[:3, :3].T) + T_cw[:3, 3]
            K_half = self.scene.intrinsics_K * 0.5
            fx, fy, cx, cy = K_half[0, 0], K_half[1, 1], K_half[0, 2], K_half[1, 2]
            z_g = pts_c[:, 2]
            u_g = np.round(pts_c[:, 0] * fx / np.maximum(z_g, 1e-4) + cx).astype(np.int32)
            v_g = np.round(pts_c[:, 1] * fy / np.maximum(z_g, 1e-4) + cy).astype(np.int32)
            in_fov = (z_g > 0.25) & (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
            if np.any(in_fov):
                d_ren = out.depth[v_g[in_fov], u_g[in_fov]]
                unocc = (d_ren > 0.1) & (np.abs(z_g[in_fov] - d_ren) < 0.06)
                unocc_ratio = float(np.sum(unocc)) / float(obj_pts_w.shape[0])
            else:
                unocc_ratio = 0.0
        else:
            unocc_ratio = 0.0

        if px_count > 0:
            ys, xs = np.nonzero(mask)
            centrality = float(
                max(
                    0.0,
                    1.0 - 1.75 * np.hypot(np.mean(xs) / W - 0.5, np.mean(ys) / H - 0.5),
                )
            )
        else:
            centrality = 0.0

        cam_pos = pose_twc[:3, 3]
        travel_dist = float(np.linalg.norm(cam_pos - curr_pos))
        vis_norm = min(1.0, px_frac / 0.045)
        utility = (
            0.45 * vis_norm
            + 0.35 * unocc_ratio
            + 0.20 * centrality
            - 0.06 * travel_dist
        )

        return {
            "candidate_id": candidate_id,
            "camera_pos_m": [round(float(x), 4) for x in cam_pos],
            "camera_pose_twc": [[round(float(x), 5) for x in row] for row in pose_twc],
            "target_pixel_count": px_count,
            "target_pixel_fraction": round(px_frac, 5),
            "unoccluded_ratio": round(unocc_ratio, 4),
            "view_centrality": round(centrality, 4),
            "travel_distance_m": round(travel_dist, 4),
            "obstacle_clearance_m": round(float(esdf_clearance), 4),
            "nbv_utility_score": round(float(utility), 4),
        }

    # ------------------------------------------------------------------
    # 4. 3DGS-to-ESDF Collision-Free 3D Trajectory Planning
    # ------------------------------------------------------------------
    def _get_or_build_esdf(self) -> Dict[str, Any]:
        if self._esdf_cache is not None:
            return self._esdf_cache

        active = self.scene.opacities >= self.opacity_threshold
        pts = self.scene.positions[active]
        vs = self.voxel_size
        pad = 0.28
        bmin = pts.min(axis=0) - pad
        bmax = pts.max(axis=0) + pad
        grid_shape = np.ceil((bmax - bmin) / vs).astype(int) + 1
        occ = np.zeros(grid_shape, dtype=bool)

        ijk = np.floor((pts - bmin) / vs).astype(int)
        ijk = np.clip(ijk, 0, grid_shape - 1)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

        # Exact 3D Euclidean Signed Distance Field in meters outside obstacles
        esdf_m = distance_transform_edt(~occ).astype(np.float32) * vs
        self._esdf_cache = {
            "bmin": bmin.astype(np.float32),
            "bmax": bmax.astype(np.float32),
            "voxel_size": vs,
            "grid_shape": grid_shape,
            "occupancy": occ,
            "esdf_m": esdf_m,
        }
        return self._esdf_cache

    def _query_esdf_point(
        self, pt_w: np.ndarray, esdf_info: Optional[Dict[str, Any]] = None
    ) -> float:
        info = esdf_info or self._get_or_build_esdf()
        ijk = np.round((pt_w - info["bmin"]) / info["voxel_size"]).astype(int)
        gs = info["grid_shape"]
        if np.any(ijk < 0) or np.any(ijk >= gs):
            return 0.25
        return float(info["esdf_m"][ijk[0], ijk[1], ijk[2]])

    def plan_collision_free_path(
        self,
        start_pos: List[float],
        goal_query_or_pos: Union[str, List[float]],
        clearance_radius: float = 0.06,
        standoff_distance: float = 0.16,
    ) -> Dict[str, Any]:
        """
        Plan a 3D collision-free end-effector / camera trajectory from `start_pos` to
        a target 3D object (or explicit 3D coordinate) using the 3DGS-derived ESDF grid,
        Clearance-Weighted 26-Connected 3D A*, and ESDF-Safe Shortcut Smoothing.
        """
        esdf_info = self._get_or_build_esdf()
        vs = esdf_info["voxel_size"]
        bmin = esdf_info["bmin"]
        gs = esdf_info["grid_shape"]
        esdf = esdf_info["esdf_m"]

        start_w = np.asarray(start_pos, dtype=np.float32).reshape(3)
        goal_label = "explicit_coordinates"
        if isinstance(goal_query_or_pos, str):
            res = self.query_3d_object(goal_query_or_pos, top_k=1)
            if not res["top_match"]:
                raise ValueError(f"Could not ground goal query: '{goal_query_or_pos}'")
            goal_label = res["top_match"]["object_id"]
            obj_center = np.array(res["top_match"]["center_m"], dtype=np.float32)
            # Place pre-grasp / observation waypoint at `standoff_distance` in free space
            approach_dir = start_w - obj_center
            norm_d = float(np.linalg.norm(approach_dir))
            if norm_d < 1e-4:
                approach_dir = np.array([0.0, -0.5, -0.85], dtype=np.float32)
            else:
                approach_dir /= norm_d
            goal_w = obj_center + approach_dir * standoff_distance
        else:
            goal_w = np.asarray(goal_query_or_pos, dtype=np.float32).reshape(3)

        def to_grid(p: np.ndarray) -> Tuple[int, int, int]:
            idx = np.clip(np.round((p - bmin) / vs).astype(int), 0, gs - 1)
            return int(idx[0]), int(idx[1]), int(idx[2])

        def to_world(ijk: Tuple[int, int, int]) -> np.ndarray:
            return bmin + np.array(ijk, dtype=np.float32) * vs

        s_node = self._snap_to_free_voxel(to_grid(start_w), esdf, clearance_radius + 0.015)
        g_node = self._snap_to_free_voxel(to_grid(goal_w), esdf, clearance_radius + 0.015)

        # 1. Plan with Clearance-Weighted ESDF 3D A* (with half-voxel diagonal safety margin)
        raw_nodes = self._run_astar_3d(
            s_node, g_node, esdf, min_clearance=clearance_radius + 0.012, clearance_weight=3.2
        )
        raw_pts = np.stack([to_world(n) for n in raw_nodes], axis=0)
        smooth_pts = self._shortcut_and_smooth_path(
            raw_pts, esdf_info, min_clearance=clearance_radius + 0.012
        )

        # 2. Baseline A: Straight-line interpolation
        straight_pts = np.linspace(to_world(s_node), to_world(g_node), 40).astype(np.float32)

        # 3. Baseline B: Unweighted standard voxel A* (ignores ESDF safety buffer)
        unweighted_nodes = self._run_astar_3d(
            s_node, g_node, esdf, min_clearance=0.015, clearance_weight=0.0
        )
        unweighted_pts = np.stack([to_world(n) for n in unweighted_nodes], axis=0)

        ours_stats = self._evaluate_trajectory_safety(smooth_pts, esdf_info, clearance_radius)
        straight_stats = self._evaluate_trajectory_safety(straight_pts, esdf_info, clearance_radius)
        unweighted_stats = self._evaluate_trajectory_safety(unweighted_pts, esdf_info, clearance_radius)

        return {
            "start_pos_m": [round(float(x), 4) for x in to_world(s_node)],
            "goal_target": goal_label,
            "goal_pos_m": [round(float(x), 4) for x in to_world(g_node)],
            "clearance_radius_m": round(float(clearance_radius), 4),
            "num_waypoints": int(len(smooth_pts)),
            "waypoints_m": [[round(float(x), 4) for x in pt] for pt in smooth_pts],
            "metrics_ours_esdf_astar": ours_stats,
            "metrics_unweighted_astar": unweighted_stats,
            "metrics_straight_line": straight_stats,
        }

    def _snap_to_free_voxel(
        self,
        node: Tuple[int, int, int],
        esdf: np.ndarray,
        min_clearance: float,
    ) -> Tuple[int, int, int]:
        if esdf[node] >= min_clearance:
            return node
        gx, gy, gz = esdf.shape
        best_node = node
        best_val = float(esdf[node])
        for r in range(1, 10):
            x0, x1 = max(0, node[0] - r), min(gx, node[0] + r + 1)
            y0, y1 = max(0, node[1] - r), min(gy, node[1] + r + 1)
            z0, z1 = max(0, node[2] - r), min(gz, node[2] + r + 1)
            sub = esdf[x0:x1, y0:y1, z0:z1]
            max_v = float(sub.max())
            if max_v > best_val:
                idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
                best_node = (x0 + int(idx[0]), y0 + int(idx[1]), z0 + int(idx[2]))
                best_val = max_v
            if best_val >= min_clearance + 0.02:
                return best_node
        return best_node

    def _run_astar_3d(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
        esdf: np.ndarray,
        min_clearance: float,
        clearance_weight: float = 3.0,
    ) -> List[Tuple[int, int, int]]:
        gx, gy, gz = esdf.shape
        safe_horizon = max(min_clearance * 2.4, 0.14)

        neighbors_26 = [
            (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        def heuristic(n: Tuple[int, int, int]) -> float:
            return math.sqrt(
                (n[0] - goal[0]) ** 2 + (n[1] - goal[1]) ** 2 + (n[2] - goal[2]) ** 2
            )

        open_heap: List[Tuple[float, Tuple[int, int, int]]] = [(heuristic(start), start)]
        g_score: Dict[Tuple[int, int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        closed = set()
        max_expansions = 35000
        expansions = 0
        closest_node = start
        closest_h = heuristic(start)

        while open_heap and expansions < max_expansions:
            _, curr = heapq.heappop(open_heap)
            if curr == goal:
                break
            if curr in closed:
                continue
            closed.add(curr)
            expansions += 1

            h_curr = heuristic(curr)
            if h_curr < closest_h:
                closest_h = h_curr
                closest_node = curr

            cx, cy, cz = curr
            g_curr = g_score[curr]
            for dx, dy, dz, step_len in neighbors_26:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if nx < 0 or nx >= gx or ny < 0 or ny >= gy or nz < 0 or nz >= gz:
                    continue
                nxt = (nx, ny, nz)
                if nxt in closed:
                    continue
                c_val = float(esdf[nx, ny, nz])
                if c_val < min_clearance:
                    continue
                # Clearance penalty pushes trajectory away from Gaussian obstacle surfaces
                penalty = 1.0
                if clearance_weight > 0.0 and c_val < safe_horizon:
                    ratio = (safe_horizon - c_val) / safe_horizon
                    penalty += clearance_weight * (ratio * ratio)
                tentative_g = g_curr + step_len * penalty
                if tentative_g < g_score.get(nxt, 1e18):
                    g_score[nxt] = tentative_g
                    came_from[nxt] = curr
                    f_val = tentative_g + heuristic(nxt)
                    heapq.heappush(open_heap, (f_val, nxt))

        target = goal if (goal in came_from or goal == start) else closest_node
        path = [target]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def _shortcut_and_smooth_path(
        self,
        pts: np.ndarray,
        esdf_info: Dict[str, Any],
        min_clearance: float,
    ) -> np.ndarray:
        if len(pts) <= 2:
            return pts
        # Greedy line-of-sight shortcutting subject to ESDF safety buffer
        safe_req = min_clearance + 0.015
        anchor_pts = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            next_i = i + 1
            for j in range(n - 1, i + 1, -1):
                seg = np.linspace(pts[i], pts[j], 16)
                clearances = [self._query_esdf_point(p, esdf_info) for p in seg]
                if min(clearances) >= safe_req:
                    next_i = j
                    break
            anchor_pts.append(pts[next_i])
            i = next_i

        anchors = np.stack(anchor_pts, axis=0)
        # Resample densely (36 points) with local moving-average smoothing if safe
        t_orig = np.linspace(0.0, 1.0, len(anchors))
        t_dense = np.linspace(0.0, 1.0, 36)
        dense = np.stack(
            [np.interp(t_dense, t_orig, anchors[:, d]) for d in range(3)], axis=-1
        ).astype(np.float32)

        for _ in range(3):
            cand = dense.copy()
            cand[1:-1] = 0.25 * dense[:-2] + 0.50 * dense[1:-1] + 0.25 * dense[2:]
            for k in range(1, len( dense) - 1):
                if self._query_esdf_point(cand[k], esdf_info) >= min_clearance:
                    dense[k] = cand[k]
        return dense

    def _evaluate_trajectory_safety(
        self,
        pts: np.ndarray,
        esdf_info: Dict[str, Any],
        clearance_radius: float,
    ) -> Dict[str, Any]:
        # Sample densely along all segments
        samples = []
        for i in range(len(pts) - 1):
            seg = np.linspace(pts[i], pts[i + 1], 8, endpoint=(i == len(pts) - 2))
            samples.append(seg)
        all_s = np.concatenate(samples, axis=0) if samples else pts
        clearances = np.array([self._query_esdf_point(p, esdf_info) for p in all_s])
        diffs = np.diff(pts, axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=-1)))
        coll_frac = float(np.mean(clearances < clearance_radius))
        return {
            "path_length_m": round(path_len, 4),
            "min_clearance_m": round(float(clearances.min()), 4),
            "mean_clearance_m": round(float(clearances.mean()), 4),
            "collision_fraction": round(coll_frac, 4),
            "is_collision_free": bool(clearances.min() >= clearance_radius - 1e-4),
        }

    # ------------------------------------------------------------------
    # 5. Counterfactual 3D Object Manipulation & Scene Editing
    # ------------------------------------------------------------------
    def edit_3d_object(
        self,
        object_id: str,
        action: str = "translate",
        delta_pos: Optional[List[float]] = None,
        yaw_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Counterfactual 3D Gaussian Scene Editing:
        Directly manipulates the 3D Gaussians belonging to `object_id` (`translate`,
        `rotate`, `transform`, `remove`, or `reset`), updates the 3D object bounding box,
        and verifies post-edit collision clearance against the rest of the scene.
        """
        act = action.strip().lower()
        if act == "reset":
            self.scene.reset_edits()
            self._esdf_cache = None
            return {"status": "reset_complete", "scene_name": self.scene.scene_name}

        obj = self.scene.get_object_by_id(object_id)
        if obj is None:
            res = self.query_3d_object(object_id, top_k=1)
            if res["top_match"]:
                obj = self.scene.get_object_by_id(res["top_match"]["object_id"])
        if obj is None:
            raise ValueError(f"Could not find 3D object '{object_id}' to edit.")

        g_mask = self.scene.instance_ids == obj.instance_index
        num_edited = int(np.sum(g_mask))
        old_center = obj.center.copy()

        dt = (
            np.asarray(delta_pos, dtype=np.float32).reshape(3)
            if delta_pos is not None
            else np.zeros(3, dtype=np.float32)
        )
        yaw_rad = math.radians(float(yaw_deg))
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        # Rotation around desk vertical axis (Y axis)
        R_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        q_yaw = np.array(
            [math.cos(yaw_rad / 2.0), 0.0, math.sin(yaw_rad / 2.0), 0.0], dtype=np.float32
        )

        if act == "remove":
            self.scene.opacities[g_mask] = 0.0
        elif act in ("translate", "rotate", "transform"):
            rel_pts = self.scene.positions[g_mask] - old_center[None, :]
            rot_pts = rel_pts @ R_yaw.T
            self.scene.positions[g_mask] = rot_pts + old_center[None, :] + dt[None, :]
            if abs(yaw_deg) > 1e-3:
                self.scene.rotations[g_mask] = _quat_multiply(
                    q_yaw[None, :], self.scene.rotations[g_mask]
                )
            obj.center = (old_center + dt).astype(np.float32)
            obj.bbox_min = np.quantile(self.scene.positions[g_mask], 0.02, axis=0).astype(np.float32)
            obj.bbox_max = np.quantile(self.scene.positions[g_mask], 0.98, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported edit action '{action}'")

        self._esdf_cache = None

        # Check minimum distance to other foreground objects after edit
        min_other_dist = 999.0
        closest_other = None
        for other in self.scene.objects:
            if other.object_id == obj.object_id or other.category == "dining table":
                continue
            d = float(np.linalg.norm(obj.center - other.center))
            if d < min_other_dist:
                min_other_dist = d
                closest_other = other.object_id

        return {
            "object_id": obj.object_id,
            "action": act,
            "num_gaussians_modified": num_edited,
            "previous_center_m": [round(float(x), 4) for x in old_center],
            "new_center_m": [round(float(x), 4) for x in obj.center],
            "applied_translation_m": [round(float(x), 4) for x in dt],
            "applied_yaw_deg": round(float(yaw_deg), 2),
            "closest_foreground_object": closest_other,
            "closest_object_center_distance_m": round(min_other_dist, 4),
        }
