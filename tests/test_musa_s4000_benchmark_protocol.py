from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import run_musa_s4000_formal_suite as formal  # noqa: E402
import stofm_musa_s4000_worker as worker  # noqa: E402


class _FakeEvent:
    def __init__(self, enable_timing: bool):
        assert enable_timing

    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert isinstance(other, _FakeEvent)
        return 2.0


class _FakeRuntime:
    Event = _FakeEvent

    @staticmethod
    def synchronize() -> None:
        pass

    @staticmethod
    def reset_peak_memory_stats() -> None:
        pass

    @staticmethod
    def memory_allocated() -> int:
        return 1024**2

    @staticmethod
    def max_memory_allocated() -> int:
        return 2 * 1024**2


def test_measurement_wraps_warmup_and_samples_in_inference_mode():
    calls = []

    def operation():
        calls.append(torch.is_inference_mode_enabled())

    result = worker._measure_inference(
        operation,
        runtime=_FakeRuntime(),
        warmup=2,
        repetitions=3,
        calls_per_sample=2,
        seed=17,
        bootstrap_resamples=100,
    )

    assert calls == [True] * 8
    assert result["sample_count"] == 3
    assert result["samples_ms"] == [1.0, 1.0, 1.0]
    assert len(result["host_samples_ms"]) == 3


def test_paired_hierarchical_bootstrap_is_deterministic_and_directional():
    pairs = [([9.8, 10.0, 10.2], [4.8, 5.0, 5.2])] * 5
    first = formal._paired_bootstrap_speedup(pairs, resamples=1000, seed=41)
    second = formal._paired_bootstrap_speedup(pairs, resamples=1000, seed=41)

    assert first == second
    assert first["point_estimate"] == pytest.approx(2.0)
    assert first["bootstrap_95_ci"][0] > 1.0


def test_primary_comparison_keeps_three_baselines_explicit():
    def row(kind: str, operator: str, torch_ms: float, flagos_ms: float):
        return {
            "candidate_kind": kind,
            "operator": operator,
            "nodes": 1050,
            "precision": "fp32",
            "torch_p50_ms": {"median": torch_ms},
            "flagos_p50_ms": {"median": flagos_ms},
            "speedup_over_torch": {"point_estimate": torch_ms / flagos_ms},
        }

    rows = []
    for operator in formal.OPERATOR_NAMES:
        rows.append(row("initial_registered_implementation", operator, 20.0, 40.0))
        rows.append(row("optimized_flagos_backend", operator, 20.0, 10.0))

    comparison = formal._primary_comparison(rows)

    assert len(comparison) == 2
    assert all(item["torch_p50_ms"] == 20.0 for item in comparison)
    assert all(item["initial_flagos_p50_ms"] == 40.0 for item in comparison)
    assert all(item["optimized_flagos_p50_ms"] == 10.0 for item in comparison)
    assert all(item["optimized_speedup_over_initial_flagos"] == 4.0 for item in comparison)
