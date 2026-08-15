import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from profile_vision_r2_aten import ATEN_CLASSIFICATION, STAGES


def test_vision_r2_profile_covers_measured_and_rejected_boundaries():
    assert set(STAGES) == {
        "marker_torch",
        "marker_nvidia",
        "swiglu_torch",
        "swiglu_nvidia",
        "residual_layer_norm_torch",
    }
    assert ATEN_CLASSIFICATION["aten::native_layer_norm"] == "torch_retained_candidate_rejected"
