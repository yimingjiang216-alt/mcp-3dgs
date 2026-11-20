"""
FastMCP Server for 3D Gaussian Splatting Spatial Intelligence (`mcp-3dgs`).

Exposes 7 standardized Model Context Protocol (MCP) tools allowing any LLM/VLM Agent
to perceive, query, imagine candidate views, plan 3D collision-free paths, and
counterfactually edit a real-world 3D Gaussian Splatting scene (`TUM fr1/desk`).

Usage:
  python -m src.mcp_server
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastmcp import FastMCP
from PIL import Image

from .gaussian_scene import SemanticGaussianScene, build_scene_from_tum_rgbd
from .renderer import look_at_pose
from .spatial_tools import SpatialIntelligenceEngine

mcp = FastMCP("mcp-3dgs-spatial-intelligence")

_ENGINE: Optional[SpatialIntelligenceEngine] = None
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCENE_NPZ = _REPO_ROOT / "results" / "tum_fr1_desk_scene.npz"
_DEFAULT_TUM_DIR = _REPO_ROOT.parent / "tum" / "extract" / "rgbd_dataset_freiburg1_desk"


def get_engine() -> SpatialIntelligenceEngine:
    """Lazy-load cached 3DGS scene or reconstruct from TUM fr1/desk."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if _DEFAULT_SCENE_NPZ.exists():
        scene = SemanticGaussianScene.load(_DEFAULT_SCENE_NPZ)
    elif _DEFAULT_TUM_DIR.exists():
        scene = build_scene_from_tum_rgbd(_DEFAULT_TUM_DIR)
        scene.save(_DEFAULT_SCENE_NPZ, _REPO_ROOT / "results" / "scene_objects.json")
    else:
        raise FileNotFoundError(
            f"Neither cached scene {_DEFAULT_SCENE_NPZ} nor TUM dataset {_DEFAULT_TUM_DIR} found."
        )
    _ENGINE = SpatialIntelligenceEngine(scene)
    return _ENGINE


@mcp.tool()
def get_scene_summary() -> Dict[str, Any]:
    """
    Return global metadata of the 3D Gaussian Splatting scene (`tum_fr1_desk`),
    including total anisotropic Gaussians, reconstructed camera trajectory length,
    3D workspace bounds, and the catalog of all discovered 3D object instances.
    """
    return get_engine().get_scene_summary()


@mcp.tool()
def query_3d_object(text_query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Ground an open-vocabulary natural language query (e.g., 'something to drink coffee from',
    'device for typing code', 'display monitor', 'reading book') to 3D Gaussian object instances
    using 384-D neural semantic embeddings and return ranked 3D bounding boxes and coordinates.
    """
    return get_engine().query_3d_object(text_query=text_query, top_k=top_k)


@mcp.tool()
def compute_spatial_relations(object_a: str, object_b: str) -> Dict[str, Any]:
    """
    Compute metric 3D center distance, minimum bounding-box surface clearance,
    directional offset vector, and topological predicates ('left_of', 'right_of',
    'in_front_of', 'adjacent_to') between two 3D objects in the scene.
    """
    return get_engine().compute_spatial_relations(object_a=object_a, object_b=object_b)


@mcp.tool()
def render_novel_view(
    camera_pos: List[float],
    look_at: List[float],
    query_text: Optional[str] = None,
    save_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Render a novel 6-DoF camera viewpoint from the 3D Gaussian Splatting scene
    using EWA covariance projection and alpha compositing. Returns visible 3D objects,
    depth statistics, and optionally saves the rendered RGB/Semantic image.
    """
    eng = get_engine()
    pose = look_at_pose(
        np.asarray(camera_pos, dtype=np.float32),
        np.asarray(look_at, dtype=np.float32),
    )
    out = eng.renderer.render(pose, scale_factor=1.0, query_text=query_text)
    saved_path = None
    if save_filename:
        out_path = _REPO_ROOT / "results" / Path(save_filename).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((out.rgb * 255).astype(np.uint8)).save(out_path)
        saved_path = str(out_path)

    valid_d = out.depth[out.alpha > 0.1]
    return {
        "camera_pos_m": [round(float(x), 4) for x in camera_pos],
        "look_at_m": [round(float(x), 4) for x in look_at],
        "resolution": [eng.renderer.width, eng.renderer.height],
        "opacity_coverage": round(float(np.mean(out.alpha > 0.1)), 4),
        "mean_depth_m": round(float(valid_d.mean()), 4) if valid_d.size else 0.0,
        "visible_objects": out.visible_objects,
        "saved_image_path": saved_path,
    }


@mcp.tool()
def imagine_candidate_views(
    target_query: str,
    current_camera_pos: Optional[List[float]] = None,
    num_candidates: int = 6,
) -> Dict[str, Any]:
    """
    Active VLN View Imagination: sample `num_candidates` collision-free camera poses
    around the queried 3D object, render all candidate views inside the 3DGS world model
    before moving the physical agent, and return the Next-Best-View maximizing visibility,
    unoccluded surface ratio, and framing centrality.
    """
    return get_engine().imagine_candidate_views(
        target_query=target_query,
        current_camera_pos=current_camera_pos,
        num_candidates=num_candidates,
    )


@mcp.tool()
def plan_collision_free_path(
    start_pos: List[float],
    goal_query_or_pos: str,
    clearance_radius: float = 0.06,
) -> Dict[str, Any]:
    """
    Plan a 3D collision-free trajectory from `start_pos` [x, y, z] to a target 3D object
    query (or '[x, y, z]' coordinate) using the 3DGS-derived Euclidean Signed Distance Field
    (ESDF), Clearance-Weighted 26-Connected 3D A*, and shortcut smoothing.
    """
    eng = get_engine()
    target: Any = goal_query_or_pos
    if goal_query_or_pos.strip().startswith("["):
        import json

        target = json.loads(goal_query_or_pos)
    return eng.plan_collision_free_path(
        start_pos=start_pos,
        goal_query_or_pos=target,
        clearance_radius=clearance_radius,
    )


@mcp.tool()
def edit_3d_object(
    object_id: str,
    action: str = "translate",
    delta_pos: Optional[List[float]] = None,
    yaw_deg: float = 0.0,
) -> Dict[str, Any]:
    """
    Counterfactual 3D Scene Editing: translate, rotate, remove, or reset the 3D Gaussians
    belonging to `object_id` inside the 3DGS scene and verify post-edit obstacle clearance.
    """
    return get_engine().edit_3d_object(
        object_id=object_id,
        action=action,
        delta_pos=delta_pos,
        yaw_deg=yaw_deg,
    )


if __name__ == "__main__":
    mcp.run()
