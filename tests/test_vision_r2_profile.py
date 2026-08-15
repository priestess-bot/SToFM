import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from profile_vision_r2_aten import ATEN_CLASSIFICATION, STAGES, classify_event


def test_vision_r2_profile_covers_measured_and_rejected_boundaries():
    assert set(STAGES) == {
        "marker_torch",
        "marker_nvidia",
        "swiglu_torch",
        "swiglu_nvidia",
        "residual_layer_norm_torch",
    }
    assert ATEN_CLASSIFICATION["aten::native_layer_norm"] == "torch_retained_candidate_rejected"
    assert classify_event("_marker_token_embed_kernel") == "nvidia_custom_marker_token_kernel"
    assert classify_event("swiglu_kernel") == "flaggems_existing_swiglu_kernel_rejected"
