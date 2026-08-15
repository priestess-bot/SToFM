import json
from pathlib import Path

import pytest

from benchmarks.aggregate_operator_runs import _bootstrap_speedup, aggregate


def _write_run(path: Path, *, p50: float, nodes: int = 17, peak_allocated_mib: float = 10.0) -> None:
    path.mkdir()
    result = {
        "run_id": path.name,
        "hardware": {"name": "test-gpu", "capability": [7, 0], "torch": "test", "cuda": "test"},
        "workload": {"nodes": nodes, "output_dir": str(path)},
        "commits": {"stofm": "a", "flaggems": "b"},
        "validation": {"status": "passed"},
        "results": [
            {
                "stage": "B0",
                "status": "measured",
                "baseline_stage": "B0",
                "p50_ms": p50,
                "peak_allocated_mib": peak_allocated_mib,
                "peak_delta_allocated_mib": 2.0,
                "samples_ms": [p50 - 0.1, p50, p50 + 0.1],
            },
            {
                "stage": "B2",
                "status": "skipped",
                "baseline_stage": "B0",
                "reason": "excluded",
                "samples_ms": [],
            },
        ],
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (path / "samples.csv").write_text("stage,sample_index,latency_ms\n", encoding="utf-8")


def test_aggregate_reports_median_checksums_and_raw_sample_counts(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_c = tmp_path / "run-c"
    _write_run(run_a, p50=3.0)
    _write_run(run_b, p50=2.0)
    _write_run(run_c, p50=4.0)

    summary = aggregate([run_a, run_b, run_c], bootstrap_resamples=100)

    assert summary["run_count"] == 3
    assert all(len(run["result_sha256"]) == 64 for run in summary["runs"])
    assert summary["stages"][0]["p50_ms"] == {"min": 2.0, "median": 3.0, "max": 4.0, "mean": 3.0}
    assert summary["stages"][0]["total_raw_samples"] == 9
    assert "speedup_vs_baseline" not in summary["stages"][0]
    assert summary["stages"][1]["reasons"] == ["excluded", "excluded", "excluded"]


def test_aggregate_rejects_mismatched_workloads(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_run(run_a, p50=3.0)
    _write_run(run_b, p50=2.0, nodes=33)

    with pytest.raises(ValueError, match="workloads differ"):
        aggregate([run_a, run_b], bootstrap_resamples=100)


def test_bootstrap_speedup_is_deterministic_and_uses_median_latency():
    result = _bootstrap_speedup([10.0, 10.0, 10.0], [5.0, 5.0, 5.0], resamples=100, seed=7)

    assert result == {
        "point_estimate": 2.0,
        "bootstrap_95_ci": {"lower": 2.0, "upper": 2.0},
        "resamples": 100,
    }


def test_aggregate_keeps_memory_unavailable_when_a_target_runtime_cannot_report_it(tmp_path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_run(run_a, p50=3.0, peak_allocated_mib=None)
    _write_run(run_b, p50=2.0, peak_allocated_mib=None)

    summary = aggregate([run_a, run_b], bootstrap_resamples=100)

    assert summary["stages"][0]["peak_allocated_mib"] is None
