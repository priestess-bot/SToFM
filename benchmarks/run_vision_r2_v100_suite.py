"""Run and aggregate independent R2 V100 Vision operator trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).parents[1]
WORKER = ROOT / "benchmarks" / "vision_r2_v100_worker.py"
DEFAULT_WORKSPACE = ROOT.parent


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "mean": statistics.mean(values)}


def _bootstrap_speedup(
    baseline: List[float], candidate: List[float], *, resamples: int, seed: int
) -> Dict[str, Any]:
    generator = random.Random(seed)
    point = statistics.median(baseline) / statistics.median(candidate)
    values = []
    for _ in range(resamples):
        sampled_baseline = [generator.choice(baseline) for _ in baseline]
        sampled_candidate = [generator.choice(candidate) for _ in candidate]
        values.append(statistics.median(sampled_baseline) / statistics.median(sampled_candidate))
    values.sort()
    return {
        "point_estimate": point,
        "bootstrap_95_ci": {
            "lower": values[int((len(values) - 1) * 0.025)],
            "upper": values[int((len(values) - 1) * 0.975)],
        },
        "resamples": resamples,
    }


def _validate_trial(result: Dict[str, Any], precision: str, workload: Dict[str, Any], references: Dict[str, str]) -> None:
    if result["precision"] != precision or result["workload"] != workload:
        raise ValueError("Vision worker precision or workload drifted between independent trials")
    if result["reference_hashes"] != references:
        raise ValueError("Vision Torch reference checksum drifted between independent trials")


def _aggregate(trials: List[Dict[str, Any]], *, resamples: int) -> Dict[str, Any]:
    rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        for row in trial["results"]:
            rows_by_stage.setdefault(row["stage"], []).append(row)
    summaries = []
    for stage, rows in sorted(rows_by_stage.items()):
        template = rows[0]
        if template["status"] != "measured":
            summaries.append(
                {
                    "stage": stage,
                    "status": template["status"],
                    "gain_kind": template["gain_kind"],
                    "comparison_baseline": template["comparison_baseline"],
                    "reason": template["reason"],
                }
            )
            continue
        if len(rows) != len(trials):
            raise ValueError(f"Vision stage {stage} is missing from an independent trial")
        summary: Dict[str, Any] = {
            "stage": stage,
            "status": "measured",
            "gain_kind": template["gain_kind"],
            "comparison_baseline": template["comparison_baseline"],
            "run_count": len(rows),
            "p50_ms": _summary([row["p50_ms"] for row in rows]),
            "total_raw_samples": sum(len(row["samples_ms"]) for row in rows),
            "peak_delta_allocated_mib": _summary([row["peak_delta_allocated_mib"] for row in rows]),
        }
        baseline_stage = template["comparison_baseline"]
        if baseline_stage != stage:
            baseline_rows = rows_by_stage.get(baseline_stage, [])
            if len(baseline_rows) != len(rows):
                raise ValueError(f"Vision stage {stage} references missing baseline {baseline_stage}")
            baseline = [sample for row in baseline_rows for sample in row["samples_ms"]]
            candidate = [sample for row in rows for sample in row["samples_ms"]]
            summary["speedup"] = _bootstrap_speedup(
                baseline, candidate, resamples=resamples, seed=20260815 + len(stage)
            )
        summaries.append(summary)
    return {"stages": summaries, "bootstrap_resamples": resamples}


def _write_report(output_dir: Path, suite: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Vision FlagOS Inference R2 V100 Suite",
        "",
        f"Precision: `{suite['precision']}`; independent trials: `{suite['run_count']}`.",
        "",
        "| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples | Status |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in suite["aggregate"]["stages"]:
        if row["status"] != "measured":
            lines.append(
                f"| {row['stage']} | {row['gain_kind']} | - | - | {row['comparison_baseline']} | - | - | 0 | {row['status']}: {row['reason']} |"
            )
            continue
        p50 = row["p50_ms"]
        speedup = row.get("speedup")
        if speedup is None:
            speedup_text, interval = "-", "-"
        else:
            speedup_text = f"{speedup['point_estimate']:.3f}x"
            interval = f"[{speedup['bootstrap_95_ci']['lower']:.3f}x, {speedup['bootstrap_95_ci']['upper']:.3f}x]"
        lines.append(
            f"| {row['stage']} | {row['gain_kind']} | {p50['median']:.4f} | {p50['min']:.4f}/{p50['max']:.4f} | "
            f"{row['comparison_baseline']} | {speedup_text} | {interval} | {row['total_raw_samples']} | measured |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--python", type=Path, default=DEFAULT_WORKSPACE / ".venv-flagos-r2" / "bin" / "python")
    parser.add_argument("--flaggems-source-root", type=Path, default=DEFAULT_WORKSPACE / "FlagGems-stofm")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--markers", type=int, default=32)
    parser.add_argument("--tokens-per-marker", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--marker-vocab", type=int, default=175)
    parser.add_argument("--swiglu-sequence", type=int, default=264)
    parser.add_argument("--swiglu-hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.runs < 3:
        raise ValueError("R2 reporting requires at least three independent processes")
    if not args.python.is_file() or not args.flaggems_source_root.is_dir():
        raise FileNotFoundError("Vision R2 requires the pinned optimized interpreter and FlagGems checkout")

    trials = []
    workload: Dict[str, Any] = {}
    references: Dict[str, str] = {}
    for run_index in range(1, args.runs + 1):
        run_dir = args.output_dir / f"run-{run_index:02d}"
        command = [
            str(args.python),
            str(WORKER),
            "--precision",
            args.precision,
            "--output-dir",
            str(run_dir),
            "--run-index",
            str(run_index),
            "--flaggems-source-root",
            str(args.flaggems_source_root),
        ]
        for name in (
            "device_index",
            "batch_size",
            "markers",
            "tokens_per_marker",
            "embedding_dim",
            "marker_vocab",
            "swiglu_sequence",
            "swiglu_hidden",
            "warmup",
            "repetitions",
            "calls_per_sample",
            "seed",
        ):
            command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "worker.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Vision worker failed ({completed.returncode}); inspect {run_dir / 'worker.log'}")
        result = _read_json(run_dir / "result.json")
        if not trials:
            workload, references = result["workload"], result["reference_hashes"]
        _validate_trial(result, args.precision, workload, references)
        trials.append(
            {
                "run_index": run_index,
                "result": str((run_dir / "result.json").relative_to(args.output_dir)),
                "result_sha256": _sha256(run_dir / "result.json"),
                "commits": result["commits"],
                "results": result["results"],
            }
        )
    suite = {
        "schema_version": 1,
        "precision": args.precision,
        "run_count": args.runs,
        "workload": workload,
        "reference_hashes": references,
        "trials": trials,
        "aggregate": _aggregate(trials, resamples=args.bootstrap_resamples),
    }
    _write_report(args.output_dir, suite)
    print(json.dumps({"output_dir": str(args.output_dir), "run_count": args.runs}, indent=2))


if __name__ == "__main__":
    main()
