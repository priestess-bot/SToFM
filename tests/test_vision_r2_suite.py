import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from run_vision_r2_v100_suite import _aggregate, _bootstrap_speedup, _validate_trial


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
