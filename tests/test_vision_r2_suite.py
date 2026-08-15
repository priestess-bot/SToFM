import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_vision_r2_v100_suite import (
    _aggregate,
    _bootstrap_speedup,
    _validate_cross_environment_trial,
    _validate_trial,
)
from vision_r2_v100_worker import STAGES


def _row(stage, baseline, samples, status="measured"):
    return {
        "stage": stage,
        "comparison_baseline": baseline,
        "gain_kind": "test",
        "status": status,
        "reason": "rejected for test" if status != "measured" else "",
        "samples_ms": samples,
        "p50_ms": sorted(samples)[len(samples) // 2] if samples else 0.0,
        "peak_delta_allocated_mib": 1.0,
    }


def test_vision_r2_bootstrap_and_rejection_rows_are_preserved():
    speedup = _bootstrap_speedup([10.0, 10.0], [5.0, 5.0], resamples=100, seed=1)
    assert speedup["point_estimate"] == 2.0
    trials = [
        {
            "results": [
                _row("V0", "V0", [10.0, 10.0]),
                _row("V1", "V0", [5.0, 5.0]),
                _row("V2", "V0", [], status="rejected"),
            ]
        }
        for _ in range(3)
    ]
    rows = {row["stage"]: row for row in _aggregate(trials, resamples=100)["stages"]}
    assert rows["V1"]["speedup"]["point_estimate"] == 2.0
    assert rows["V2"]["status"] == "rejected"


def test_vision_r2_rejects_reference_drift_between_workers():
    result = {"precision": "fp32", "workload": {"markers": 32}, "reference_hashes": {"marker": "changed"}}
    try:
        _validate_trial(result, "fp32", {"markers": 32}, {"marker": "expected"})
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("expected reference checksum drift to be rejected")


def test_vision_r2_requires_comparable_explicit_stock_and_optimized_workers():
    shared = {
        "precision": "fp32",
        "workload": {"markers": 32},
        "reference_hashes": {"marker": "expected"},
        "measurement": {"warmup": 10, "repetitions": 30},
        "runtime": {
            "torch": "2.6.0+cu124",
            "cuda": "12.4",
            "device": "Tesla V100-SXM2-16GB",
            "capability": [7, 0],
            "torch_backend": {"allow_tf32_matmul": False},
        },
    }
    stock = {**shared, "role": "stock"}
    optimized = {**shared, "role": "optimized"}
    _validate_cross_environment_trial(stock, optimized, "fp32", {"markers": 32}, {"marker": "expected"})

    drifted = {**optimized, "runtime": {**optimized["runtime"], "cuda": "12.5"}}
    try:
        _validate_cross_environment_trial(stock, drifted, "fp32", {"markers": 32}, {"marker": "expected"})
    except ValueError as exc:
        assert "runtime drifted" in str(exc)
    else:
        raise AssertionError("expected cross-environment runtime drift to be rejected")


def test_vision_candidates_are_compared_against_the_frozen_flagos_stages():
    candidate_baselines = {
        stage.name: stage.comparison_baseline
        for stage in STAGES
        if stage.name in {"V1_marker_token_nvidia", "V3_swiglu_nvidia", "V5_residual_layer_norm_rejected"}
    }
    assert candidate_baselines == {
        "V1_marker_token_nvidia": "V0s_marker_token_stock_flagos",
        "V3_swiglu_nvidia": "V2s_swiglu_stock_flagos",
        "V5_residual_layer_norm_rejected": "V4s_residual_layer_norm_stock_flagos",
    }
