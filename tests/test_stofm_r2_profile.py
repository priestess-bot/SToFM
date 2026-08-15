import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from profile_stofm_r2_aten import classify_event


def test_stofm_profile_classification_is_stage_aware():
    assert classify_event("p1", "aten::addmm") == "torch_reference_aten"
    assert classify_event("f0", "aten::addmm") == "stock_flagos_aten"
    assert classify_event("final", "aten::addmm") == "optimized_flagos_aten"
    assert classify_event("final", "triton_poi_fused_abs_add_div_exp_mul_pow_sub_0") == "gaussian_compiler_kernel"
    assert classify_event("f0", "aten::native_layer_norm") == "torch_retained_candidate_rejected"
