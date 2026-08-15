"""Aggregate independently generated operator benchmark result files.

The tool deliberately validates run compatibility before calculating summary
statistics. It can be reused for V100, Ascend, MTT, and future vision shape
sweeps without changing the raw benchmark format.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(run_dir: Path) -> Dict[str, Any]:
    result_path = run_dir / "result.json"
    samples_path = run_dir / "samples.csv"
    if not result_path.is_file() or not samples_path.is_file():
        raise ValueError(f"{run_dir} must contain result.json and samples.csv")
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result.get("results"), list):
        raise ValueError(f"{result_path} has no results list")
    return {
        "path": str(run_dir),
        "result": result,
        "result_sha256": _sha256(result_path),
        "samples_sha256": _sha256(samples_path),
    }


def _workload_for_comparison(workload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in workload.items() if key != "output_dir"}


def _hardware_for_comparison(hardware: Dict[str, Any]) -> Dict[str, Any]:
    return {key: hardware.get(key) for key in ("name", "capability", "torch", "cuda")}


def _stage_contract(result: Dict[str, Any]) -> List[tuple]:
    return [
        (row.get("stage"), row.get("status"), row.get("baseline_stage"))
        for row in result["results"]
    ]


def _stats(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def _optional_stats(values: Iterable[Any]) -> Optional[Dict[str, float]]:
    values = list(values)
    if any(value is None for value in values):
        return None
    return _stats(float(value) for value in values)


def _quantile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _bootstrap_speedup(
    baseline_samples: List[float],
    candidate_samples: List[float],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Return a deterministic non-parametric CI for median-latency speedup."""
    if not baseline_samples or not candidate_samples:
        raise ValueError("Measured stages must include raw samples")
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        baseline = statistics.median(
            baseline_samples[rng.randrange(len(baseline_samples))] for _ in baseline_samples
        )
        candidate = statistics.median(
            candidate_samples[rng.randrange(len(candidate_samples))] for _ in candidate_samples
        )
        ratios.append(baseline / candidate)
    return {
        "point_estimate": statistics.median(baseline_samples) / statistics.median(candidate_samples),
        "bootstrap_95_ci": {"lower": _quantile(ratios, 0.025), "upper": _quantile(ratios, 0.975)},
        "resamples": resamples,
    }


def aggregate(
    run_dirs: Iterable[Path], *, bootstrap_resamples: int = 10_000, bootstrap_seed: int = 20260815
) -> Dict[str, Any]:
    """Validate compatible result directories and calculate per-stage summaries."""
    runs = [_load_run(Path(run_dir)) for run_dir in run_dirs]
    if len(runs) < 2:
        raise ValueError("At least two independent benchmark runs are required")

    first = runs[0]["result"]
    first_workload = _workload_for_comparison(first.get("workload", {}))
    first_hardware = _hardware_for_comparison(first.get("hardware", {}))
    first_commits = first.get("commits")
    first_contract = _stage_contract(first)
    for run in runs[1:]:
        result = run["result"]
        if _workload_for_comparison(result.get("workload", {})) != first_workload:
            raise ValueError("Benchmark workloads differ")
        if _hardware_for_comparison(result.get("hardware", {})) != first_hardware:
            raise ValueError("Benchmark hardware/runtime differs")
        if result.get("commits") != first_commits:
            raise ValueError("Benchmark implementation commits differ")
        if _stage_contract(result) != first_contract:
            raise ValueError("Benchmark stage contracts differ")

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")

    raw_samples_by_stage = {
        stage_name: [
            float(sample)
            for run in runs
            for sample in run["result"]["results"][stage_index].get("samples_ms", [])
        ]
        for stage_index, (stage_name, status, _) in enumerate(first_contract)
        if status == "measured"
    }
    stages = []
    for stage_index, (stage_name, status, baseline_stage) in enumerate(first_contract):
        rows = [run["result"]["results"][stage_index] for run in runs]
        summary: Dict[str, Any] = {
            "stage": stage_name,
            "status": status,
            "baseline_stage": baseline_stage,
        }
        if status == "measured":
            p50_values = [float(row["p50_ms"]) for row in rows]
            peak_values = [row.get("peak_allocated_mib") for row in rows]
            peak_delta_values = [row.get("peak_delta_allocated_mib") for row in rows]
            sample_counts = [len(row.get("samples_ms", [])) for row in rows]
            summary.update(
                {
                    "run_p50_ms": p50_values,
                    "p50_ms": _stats(p50_values),
                    "peak_allocated_mib": _optional_stats(peak_values),
                    "peak_delta_allocated_mib": _optional_stats(peak_delta_values),
                    "samples_per_run": sample_counts,
                    "total_raw_samples": sum(sample_counts),
                }
            )
            if baseline_stage != stage_name:
                summary["speedup_vs_baseline"] = _bootstrap_speedup(
                    raw_samples_by_stage[baseline_stage],
                    raw_samples_by_stage[stage_name],
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + stage_index,
                )
        else:
            summary["reasons"] = [row.get("reason") for row in rows]
        stages.append(summary)

    return {
        "schema_version": 1,
        "run_count": len(runs),
        "bootstrap": {"resamples": bootstrap_resamples, "seed": bootstrap_seed},
        "runs": [
            {
                "path": run["path"],
                "run_id": run["result"].get("run_id"),
                "result_sha256": run["result_sha256"],
                "samples_sha256": run["samples_sha256"],
            }
            for run in runs
        ],
        "common": {
            "commits": first_commits,
            "hardware": first_hardware,
            "workload": first_workload,
            "validation": first.get("validation"),
        },
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    args = parser.parse_args()
    summary = aggregate(
        args.run_dirs,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
