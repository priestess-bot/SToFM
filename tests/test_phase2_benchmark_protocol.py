from pathlib import Path
import sys

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_stofm_phase2_v100_suite import (
    _hierarchical_speedup,
    _mapping_errors,
)
from benchmarks.stofm_phase2_v100_worker import ROUTES


def test_phase2_routes_keep_framework_operator_and_optimizer_attribution_separate():
    assert set(ROUTES) == {
        "torch_scalar",
            "torch_fused",
            "torch_compile_fused",
        "flagos_reference_scalar",
        "flagos_native_scalar",
        "flagos_reference_fused",
        "flagos_native_fused",
        "flagos_vendor_reference_scalar",
        "flagos_vendor_native_scalar",
        "flagos_vendor_native_fused",
        "flagos_vendor_native_fused_v100_tuned",
    }
    assert ROUTES["flagos_reference_scalar"]["training_implementation"] == "reference"
    assert ROUTES["flagos_native_scalar"]["training_implementation"] == "native"
    assert ROUTES["flagos_reference_fused"]["optimizer"] == "flagos_fused"
    assert ROUTES["torch_compile_fused"]["framework"] == "torch_compile"
    assert ROUTES["flagos_native_fused"]["optimizer"] == "flagos_fused"
    assert ROUTES["flagos_vendor_native_fused"]["gemm_backend"] == "vendor"
    assert ROUTES["flagos_vendor_native_fused_v100_tuned"]["aten_include"] == ()
    assert all("F0" not in route["display_name"] for route in ROUTES.values())


def test_hierarchical_bootstrap_is_deterministic_and_uses_baseline_over_candidate():
    baseline = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    candidate = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    first = _hierarchical_speedup(baseline, candidate, resamples=100, seed=3)
    second = _hierarchical_speedup(baseline, candidate, resamples=100, seed=3)
    assert first == second
    assert first["speedup"] == 2.0
    assert first["bootstrap_95_ci"] == {"lower": 2.0, "upper": 2.0}


def test_correctness_mapping_reports_worst_tensor_and_absolute_error():
    reference = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    actual = {"a": torch.tensor([1.0, 2.25]), "b": torch.tensor([3.1])}
    result = _mapping_errors(actual, reference)
    assert result["tensor_count"] == 2
    assert result["worst_abs_tensor"] == "a"
    assert abs(result["max_abs"] - 0.25) < 1e-7
