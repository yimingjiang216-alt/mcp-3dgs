"""
MCP-3DGS: Model Context Protocol Server for 3D Gaussian Splatting Spatial Intelligence.

Bridges real-world 3D Gaussian Splatting scenes (reconstructed from TUM RGB-D fr1/desk
and bundle-adjusted SLAM poses) with LLM/VLM Embodied Agents via FastMCP tools:
  1. Open-vocabulary 3D object grounding (Mask R-CNN v2 2D-to-3D lifting + 384-D neural language field)
  2. Multi-modal novel-view rendering (EWA covariance splatting for RGB, Metric Depth, Semantic & Language Heatmaps)
  3. Metric & topological 3D spatial relation reasoning
  4. Active VLN candidate-view imagination (Next-Best-View selection before physical motion)
  5. 3DGS-to-ESDF collision-free 3D trajectory planning
  6. Counterfactual 3D object manipulation & scene editing
"""

from .gaussian_scene import (
    LocalSemanticEncoder,
    Object3DInstance,
    SemanticGaussianScene,
    build_scene_from_tum_rgbd,
)
from .renderer import GaussianRenderer, RenderOutput
from .spatial_tools import SpatialIntelligenceEngine

__all__ = [
    "LocalSemanticEncoder",
    "Object3DInstance",
    "SemanticGaussianScene",
    "build_scene_from_tum_rgbd",
    "GaussianRenderer",
    "RenderOutput",
    "SpatialIntelligenceEngine",
]
