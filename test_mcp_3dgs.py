"""
Comprehensive Unit & Integration Test Suite for MCP-3DGS (`test_mcp_3dgs.py`).

Verifies:
  1. Semantic 3D Gaussian Scene invariants (positive scales, unit quaternions, 3D covariance SPD)
  2. EWA 3D-to-2D covariance projection & multi-modal rendering (RGB, Metric Depth, Semantic, Language Heatmap)
  3. Open-Vocabulary 3D Object Grounding (`query_3d_object`) with 384-D neural embeddings
  4. Metric & Topological 3D Spatial Relation Reasoning (`compute_spatial_relations`)
  5. Active VLN Candidate-View Imagination (`imagine_candidate_views`)
  6. 3DGS-to-ESDF Collision-Free 3D Path Planning (`plan_collision_free_path`)
  7. Counterfactual 3D Object Manipulation (`edit_3d_object`) & FastMCP Tool Server Invocation
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gaussian_scene import SemanticGaussianScene
from src.renderer import GaussianRenderer, look_at_pose
from src.spatial_tools import SpatialIntelligenceEngine
import src.mcp_server as mcp_srv


def get_test_engine() -> SpatialIntelligenceEngine:
    return mcp_srv.get_engine()


def test_1_gaussian_scene_invariants() -> None:
    eng = get_test_engine()
    scene = eng.scene
    assert scene.num_gaussians > 10000, f"Expected >10k Gaussians, got {scene.num_gaussians}"
    assert len(scene.objects) >= 6, f"Expected >=6 3D objects, got {len(scene.objects)}"
    assert np.all(scene.scales > 0.0), "Gaussian scales must be strictly positive"
    assert np.all((scene.opacities > 0.0) & (scene.opacities <= 1.0)), "Opacities must be in (0, 1]"

    q_norms = np.linalg.norm(scene.rotations, axis=-1)
    assert np.allclose(q_norms, 1.0, atol=1e-4), "Quaternions must be unit length"

    # Check 3D covariance symmetry and positive-definiteness on sample
    cov3d = scene.get_covariances_3d()[:200]
    assert np.allclose(cov3d, np.swapaxes(cov3d, -1, -2), atol=1e-5), "3D covariances must be symmetric"
    eigvals = np.linalg.eigvalsh(cov3d)
    assert np.all(eigvals > 0.0), "3D covariance matrices must be positive definite"
    print(f"[PASS] Test 1: Scene invariants verified ({scene.num_gaussians:,} Gaussians, {len(scene.objects)} 3D objects)")


def test_2_ewa_multimodal_renderer() -> None:
    eng = get_test_engine()
    pose = eng.scene.camera_poses_twc[40]
    out = eng.renderer.render(pose, scale_factor=0.5, query_text="computer keyboard")
    assert out.rgb.shape == (240, 320, 3)
    assert out.depth.shape == (240, 320)
    assert out.instance_map.shape == (240, 320)
    assert out.relevancy_map is not None and out.relevancy_map.shape == (240, 320)
    assert float(np.mean(out.alpha > 0.1)) > 0.50, "Expected >50% scene opacity coverage"
    assert len(out.visible_objects) >= 2, "Expected multiple visible 3D objects in keyframe #40"
    print(f"[PASS] Test 2: EWA multi-modal renderer verified ({len(out.visible_objects)} visible objects)")


def test_3_open_vocabulary_3d_grounding() -> None:
    eng = get_test_engine()
    r_mug = eng.query_3d_object("something to drink hot coffee from", top_k=3)
    assert r_mug["top_match"] is not None
    assert r_mug["top_match"]["category"] == "cup", f"Expected cup, got {r_mug['top_match']}"

    r_kb = eng.query_3d_object("mechanical device for typing code", top_k=3)
    assert r_kb["top_match"] is not None
    assert r_kb["top_match"]["category"] == "keyboard", f"Expected keyboard, got {r_kb['top_match']}"
    print("[PASS] Test 3: Open-vocabulary 3D object grounding verified")


def test_4_spatial_relations() -> None:
    eng = get_test_engine()
    rel = eng.compute_spatial_relations("computer keyboard", "coffee mug")
    assert rel["center_distance_m"] > 0.05
    assert len(rel["predicates"]) >= 1
    assert isinstance(rel["natural_language_summary"], str)
    print(f"[PASS] Test 4: Spatial relations verified ({rel['natural_language_summary']})")


def test_5_vln_candidate_view_imagination() -> None:
    eng = get_test_engine()
    imag = eng.imagine_candidate_views("coffee mug for drinking", num_candidates=5)
    ig = imag["information_gain_summary"]
    assert ig["utility_score_nbv"] >= ig["utility_score_initial"], (
        "Imagined NBV must have utility >= initial view"
    )
    assert ig["visibility_fraction_nbv"] > 0.002, "NBV must have non-zero target visibility"
    print(
        f"[PASS] Test 5: VLN view imagination verified "
        f"(NBV utility={ig['utility_score_nbv']:.3f} vs initial={ig['utility_score_initial']:.3f})"
    )


def test_6_esdf_collision_free_path_planning() -> None:
    eng = get_test_engine()
    plan = eng.plan_collision_free_path(
        start_pos=[0.40, 0.15, 0.85],
        goal_query_or_pos="coffee mug",
        clearance_radius=0.055,
    )
    m_ours = plan["metrics_ours_esdf_astar"]
    assert m_ours["is_collision_free"], f"Expected collision-free path, got {m_ours}"
    assert m_ours["min_clearance_m"] >= 0.054
    assert plan["num_waypoints"] >= 10
    print(
        f"[PASS] Test 6: 3DGS-ESDF path planning verified "
        f"(min_clearance={m_ours['min_clearance_m']*100:.2f} cm, length={m_ours['path_length_m']:.3f} m)"
    )


def test_7_counterfactual_editing_and_mcp_tools() -> None:
    eng = get_test_engine()
    orig_obj = eng.scene.get_object_by_id("coffee_mug")
    assert orig_obj is not None
    orig_c = orig_obj.center.copy()

    edit_out = mcp_srv.edit_3d_object(
        object_id="coffee_mug",
        action="transform",
        delta_pos=[0.15, -0.05, -0.08],
        yaw_deg=25.0,
    )
    new_c = np.array(edit_out["new_center_m"], dtype=np.float32)
    assert np.allclose(new_c - orig_c, np.array([0.15, -0.05, -0.08]), atol=1e-2)

    # Reset and verify restoration
    mcp_srv.edit_3d_object(object_id="coffee_mug", action="reset")
    assert np.allclose(orig_obj.center, orig_c, atol=1e-4)

    summary = mcp_srv.get_scene_summary()
    assert summary["num_gaussians"] == eng.scene.num_gaussians
    print("[PASS] Test 7: Counterfactual 3D scene editing & FastMCP tools verified")


if __name__ == "__main__":
    test_1_gaussian_scene_invariants()
    test_2_ewa_multimodal_renderer()
    test_3_open_vocabulary_3d_grounding()
    test_4_spatial_relations()
    test_5_vln_candidate_view_imagination()
    test_6_esdf_collision_free_path_planning()
    test_7_counterfactual_editing_and_mcp_tools()
    print("\nALL 7 MCP-3DGS TESTS PASSED!")
