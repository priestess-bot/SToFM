import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from stofm_r2_v100_worker import _stage_specs
from run_stofm_r2_v100_suite import _aggregate, _bootstrap_speedup, _validate_trial


def _row(stage, baseline, samples, gain_kind="test"):
    return {
        "stage": stage,
        "comparison_baseline": baseline,
        "gain_kind": gain_kind,
        "samples_ms": samples,
        "p50_ms": sorted(samples)[len(samples) // 2],
        "peak_delta_allocated_mib": 1.0,
    }


def _trial(index):
    return {
        "run_index": index,
        "results": [
            _row("P1_canonical_torch", "P1_canonical_torch", [10.0, 11.0, 12.0]),
            _row("F0_stock_steady", "P1_canonical_torch", [8.0, 9.0, 10.0], "stock_aten"),
            _row("Ffinal_optimized_steady", "F0_stock_steady", [4.0, 5.0, 6.0], "combined"),
        ],
    }


def test_bootstrap_speedup_is_deterministic_for_r2_reports():
    result = _bootstrap_speedup([10.0, 10.0, 10.0], [5.0, 5.0, 5.0], resamples=100, seed=1)
    assert result == {
        "point_estimate": 2.0,
        "bootstrap_95_ci": {"lower": 2.0, "upper": 2.0},
        "resamples": 100,
    }


def test_aggregate_keeps_stock_and_combined_baselines_separate():
    result = _aggregate([_trial(1), _trial(2), _trial(3)], bootstrap_resamples=100)
    rows = {row["stage"]: row for row in result["stages"]}
    assert rows["P1_canonical_torch"]["total_raw_samples"] == 9
    assert rows["F0_stock_steady"]["comparison_baseline"] == "P1_canonical_torch"
    assert rows["Ffinal_optimized_steady"]["comparison_baseline"] == "F0_stock_steady"
    assert rows["Ffinal_optimized_steady"]["speedup"]["point_estimate"] == 1.8


def test_validate_trial_rejects_cross_environment_torch_semantic_drift():
    stock = {
        "precision": "fp32",
        "workload": {"nodes": 33},
        "reference": {"last_hidden_state_sha256": "a"},
        "role": "stock",
    }
    optimized = {
        "precision": "fp32",
        "workload": {"nodes": 33},
        "reference": {"last_hidden_state_sha256": "b"},
        "role": "optimized",
    }
    try:
        _validate_trial(stock, optimized)
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("expected semantic drift to be rejected")


def test_registered_operator_suite_separates_new_ops_from_aten_dispatch():
    stock = _stage_specs("stock", "registered_ops")
    optimized = _stage_specs("optimized", "registered_ops")

    assert [stage.name for stage in stock] == [
        "pure_pytorch_reference",
        "unoptimized_flagos_lifecycle",
        "unoptimized_flagos_steady",
    ]
    assert {stage.name for stage in optimized} == {
        "gaussian_registered_operator_only",
        "pair_score_registered_operator_only",
        "registered_operators_only_combined",
        "registered_operators_with_flagos_aten_steady",
        "registered_operators_with_flagos_aten_lifecycle",
    }
    isolated = [stage for stage in optimized if stage.name.endswith("operator_only")]
    assert isolated
    assert all(not stage.aten_dispatch for stage in isolated)
    assert all(stage.expected_custom_ops for stage in isolated)
    assert all(stage.gaussian_backend != "inductor" for stage in optimized)
