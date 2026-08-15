"""Run and aggregate independent frozen-stock and optimized SToFM R2 trials."""

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
WORKER = ROOT / "benchmarks" / "stofm_r2_v100_worker.py"
DEFAULT_WORKSPACE = ROOT.parent


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def _bootstrap_speedup(
    baseline: List[float], candidate: List[float], *, resamples: int, seed: int
) -> Dict[str, Any]:
    if not baseline or not candidate:
        raise ValueError("bootstrap requires non-empty baseline and candidate samples")
    generator = random.Random(seed)
    point = statistics.median(baseline) / statistics.median(candidate)
    values = []
    for _ in range(resamples):
        sampled_baseline = [generator.choice(baseline) for _ in baseline]
        sampled_candidate = [generator.choice(candidate) for _ in candidate]
        values.append(statistics.median(sampled_baseline) / statistics.median(sampled_candidate))
    values.sort()
    lower = values[int((len(values) - 1) * 0.025)]
    upper = values[int((len(values) - 1) * 0.975)]
    return {
        "point_estimate": point,
        "bootstrap_95_ci": {"lower": lower, "upper": upper},
        "resamples": resamples,
    }


def _worker_command(
    python: Path,
    *,
    role: str,
    precision: str,
    output_dir: Path,
    run_index: int,
    flaggems_source_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        str(python),
        str(WORKER),
        "--role",
        role,
        "--suite",
        args.suite,
        "--precision",
        precision,
        "--output-dir",
        str(output_dir),
        "--run-index",
        str(run_index),
        "--flaggems-source-root",
        str(flaggems_source_root),
    ]
    for name in (
        "device_index",
        "batch_size",
        "nodes",
        "layers",
        "embedding_dim",
        "ffn_embedding_dim",
        "heads",
        "gaussian_hidden_dim",
        "input_dim",
        "warmup",
        "repetitions",
        "calls_per_sample",
        "seed",
    ):
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    return command


def _run_worker(command: List[str], log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"worker failed ({completed.returncode}); inspect {log_path}:\n{completed.stdout[-4000:]}"
        )


def _validate_trial(stock: Dict[str, Any], optimized: Dict[str, Any]) -> None:
    if stock["precision"] != optimized["precision"]:
        raise ValueError("stock and optimized precision differ")
    if stock["workload"] != optimized["workload"]:
        raise ValueError("stock and optimized workloads differ")
    if stock["reference"]["last_hidden_state_sha256"] != optimized["reference"]["last_hidden_state_sha256"]:
        raise ValueError("pure Torch P1 reference checksum differs between stock and optimized environments")
    if stock["role"] != "stock" or optimized["role"] != "optimized":
        raise ValueError("unexpected worker role")
    stock_suite = stock.get("benchmark_suite", "legacy")
    optimized_suite = optimized.get("benchmark_suite", "legacy")
    if stock_suite != optimized_suite:
        raise ValueError("stock and optimized benchmark suites differ")
    for result in (stock, optimized):
        source = Path(result["flaggems_source"]["source_path"]).resolve()
        imported = Path(result["flaggems_source"]["imported_package"]).resolve()
        if not imported.is_relative_to(source):
            raise ValueError("worker imported FlagGems from outside its requested source root")


def _aggregate(trials: List[Dict[str, Any]], *, bootstrap_resamples: int) -> Dict[str, Any]:
    rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        for row in trial["results"]:
            rows_by_stage.setdefault(row["stage"], []).append(row)
    stage_summaries = []
    for stage, rows in sorted(rows_by_stage.items()):
        if len(rows) != len(trials):
            raise ValueError(f"stage {stage} is missing from an independent trial")
        p50s = [row["p50_ms"] for row in rows]
        summary: Dict[str, Any] = {
            "stage": stage,
            "gain_kind": rows[0]["gain_kind"],
            "comparison_baseline": rows[0]["comparison_baseline"],
            "run_count": len(rows),
            "p50_ms": _summary(p50s),
            "total_raw_samples": sum(len(row["samples_ms"]) for row in rows),
            "peak_delta_allocated_mib": _summary(
                [row["peak_delta_allocated_mib"] for row in rows]
            ),
        }
        baseline_stage = rows[0]["comparison_baseline"]
        if baseline_stage != stage:
            baseline_rows = rows_by_stage.get(baseline_stage)
            if baseline_rows is None or len(baseline_rows) != len(rows):
                raise ValueError(f"stage {stage} references missing baseline {baseline_stage}")
            baseline_samples = [value for row in baseline_rows for value in row["samples_ms"]]
            candidate_samples = [value for row in rows for value in row["samples_ms"]]
            summary["speedup"] = _bootstrap_speedup(
                baseline_samples,
                candidate_samples,
                resamples=bootstrap_resamples,
                seed=20260815 + len(stage),
            )
        stage_summaries.append(summary)
    return {"stages": stage_summaries, "bootstrap_resamples": bootstrap_resamples}


def _write_report(output_dir: Path, suite: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SToFM FlagOS Inference R2 V100 Suite",
        "",
        f"Precision: `{suite['precision']}`; independent trials: `{suite['run_count']}`.",
        "",
        "| Stage | Gain kind | p50 median ms | p50 min/max ms | Baseline | Speedup | 95% bootstrap CI | Raw samples |",
        "| --- | --- | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for row in suite["aggregate"]["stages"]:
        p50 = row["p50_ms"]
        speedup = row.get("speedup")
        if speedup is None:
            speedup_text = "-"
            interval = "-"
        else:
            speedup_text = f"{speedup['point_estimate']:.3f}x"
            interval = (
                f"[{speedup['bootstrap_95_ci']['lower']:.3f}x, "
                f"{speedup['bootstrap_95_ci']['upper']:.3f}x]"
            )
        lines.append(
            f"| {row['stage']} | {row['gain_kind']} | {p50['median']:.4f} | "
            f"{p50['min']:.4f}/{p50['max']:.4f} | {row['comparison_baseline']} | "
            f"{speedup_text} | {interval} | {row['total_raw_samples']} |"
        )
    lines.extend(
        [
            "",
        "The fixed-version unoptimized FlagOS result is collected in a separate process "
        "and package environment. Compiler, registered custom-operator, and "
        "scope-lifecycle stages are not conflated.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["legacy", "registered_ops"],
        default="legacy",
        help="Use registered_ops for the post-correction custom-operator evidence suite.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--stock-python",
        type=Path,
        default=DEFAULT_WORKSPACE / ".venv-flagos-stock-r2" / "bin" / "python",
    )
    parser.add_argument(
        "--optimized-python",
        type=Path,
        default=DEFAULT_WORKSPACE / ".venv-flagos-r2" / "bin" / "python",
    )
    parser.add_argument(
        "--stock-flaggems-root",
        type=Path,
        default=DEFAULT_WORKSPACE / "FlagGems-stock-r2",
    )
    parser.add_argument(
        "--optimized-flaggems-root",
        type=Path,
        default=DEFAULT_WORKSPACE / "FlagGems-stofm",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ffn-embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.runs < 3:
        raise ValueError("R2 reporting requires at least three independent processes")
    for executable in (args.stock_python, args.optimized_python):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    for source_root in (args.stock_flaggems_root, args.optimized_flaggems_root):
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)

    trials = []
    for run_index in range(1, args.runs + 1):
        trial_dir = args.output_dir / f"run-{run_index:02d}"
        stock_dir = trial_dir / "stock"
        optimized_dir = trial_dir / "optimized"
        _run_worker(
            _worker_command(
                args.stock_python,
                role="stock",
                precision=args.precision,
                output_dir=stock_dir,
                run_index=run_index,
                flaggems_source_root=args.stock_flaggems_root,
                args=args,
            ),
            stock_dir / "worker.log",
        )
        _run_worker(
            _worker_command(
                args.optimized_python,
                role="optimized",
                precision=args.precision,
                output_dir=optimized_dir,
                run_index=run_index,
                flaggems_source_root=args.optimized_flaggems_root,
                args=args,
            ),
            optimized_dir / "worker.log",
        )
        stock = _read_json(stock_dir / "result.json")
        optimized = _read_json(optimized_dir / "result.json")
        _validate_trial(stock, optimized)
        trials.append(
            {
                "run_index": run_index,
                "stock_result": str((stock_dir / "result.json").relative_to(args.output_dir)),
                "stock_sha256": _sha256(stock_dir / "result.json"),
                "optimized_result": str((optimized_dir / "result.json").relative_to(args.output_dir)),
                "optimized_sha256": _sha256(optimized_dir / "result.json"),
                "results": stock["results"] + optimized["results"],
                "reference_sha256": stock["reference"]["last_hidden_state_sha256"],
                "commits": {
                    "stofm": stock["commits"]["stofm"],
                    "stock_flaggems": stock["commits"]["flaggems"],
                    "optimized_flaggems": optimized["commits"]["flaggems"],
                },
            }
        )

    workload = _read_json(args.output_dir / "run-01" / "stock" / "result.json")["workload"]
    suite = {
        "schema_version": 3,
        "benchmark_suite": args.suite,
        "precision": args.precision,
        "run_count": args.runs,
        "workload": workload,
        "trials": trials,
        "aggregate": _aggregate(trials, bootstrap_resamples=args.bootstrap_resamples),
    }
    _write_report(args.output_dir, suite)
    print(json.dumps({"output_dir": str(args.output_dir), "run_count": args.runs}, indent=2))


if __name__ == "__main__":
    main()
