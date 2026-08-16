#!/usr/bin/env python3
"""Run and aggregate the correctness-gated MTT S4000 evidence suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
MATRIX_WORKER = ROOT / "benchmarks" / "stofm_musa_s4000_operator_matrix.py"
MODEL_WORKER = ROOT / "benchmarks" / "stofm_musa_s4000_worker.py"
STOCK_PROBE = ROOT / "benchmarks" / "probe_musa_stock_flagos.py"

OPERATOR_NAMES = {
    "gaussian_pair_bias": "Gaussian pair-bias",
    "pair_score_epilogue": "Pair-attention score/softmax/context",
}
STAGE_NAMES = {
    "pure_pytorch": "Pure PyTorch inference",
    "native_gaussian_pair_bias": "FlagOS Gaussian optimization only",
    "native_pair_score_epilogue": "FlagOS pair-attention optimization only",
    "native_stofm_operators_combined": "Both optimized FlagOS operators",
    "native_stofm_operators_plus_flagos_aten": "Both operators plus generic FlagOS ATen dispatch",
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": mean,
        "stdev": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _paired_bootstrap_speedup(
    sample_pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap trial blocks and samples within each selected block."""
    if not sample_pairs or resamples <= 0:
        raise ValueError("paired bootstrap requires trials and positive resamples")
    baseline = [value for pair in sample_pairs for value in pair[0]]
    candidate = [value for pair in sample_pairs for value in pair[1]]
    point = statistics.median(baseline) / statistics.median(candidate)
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sampled_baseline: List[float] = []
        sampled_candidate: List[float] = []
        for _ in sample_pairs:
            baseline_values, candidate_values = generator.choice(sample_pairs)
            sampled_baseline.extend(
                generator.choice(baseline_values) for _ in baseline_values
            )
            sampled_candidate.extend(
                generator.choice(candidate_values) for _ in candidate_values
            )
        estimates.append(
            statistics.median(sampled_baseline) / statistics.median(sampled_candidate)
        )
    return {
        "point_estimate": point,
        "bootstrap_95_ci": [
            _quantile(estimates, 0.025),
            _quantile(estimates, 0.975),
        ],
        "resamples": resamples,
        "method": "paired hierarchical bootstrap over trial blocks and raw device-event samples",
    }


def _run(command: List[str], *, log_path: Path, accepted_codes: Tuple[int, ...] = (0,)) -> int:
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
    if completed.returncode not in accepted_codes:
        raise RuntimeError(
            f"command failed with {completed.returncode}; inspect {log_path}:\n"
            f"{completed.stdout[-4000:]}"
        )
    return completed.returncode


def _matrix_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    trial: int,
    candidate: str,
    library: Path,
    shapes: str,
    precisions: str,
    warmup: int,
    repetitions: int,
) -> List[str]:
    return [
        str(args.python),
        str(MATRIX_WORKER),
        "--output-dir",
        str(output_dir),
        "--flaggems-root",
        str(args.flaggems_root),
        "--library",
        str(library),
        "--candidate",
        candidate,
        "--shapes",
        shapes,
        "--precisions",
        precisions,
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
        "--bootstrap-resamples",
        str(args.worker_bootstrap_resamples),
        "--trial",
        str(trial),
        "--seed",
        str(args.seed),
        "--stofm-revision",
        args.stofm_revision,
        "--flaggems-revision",
        args.flaggems_revision,
    ]


def _model_command(args: argparse.Namespace, *, output_dir: Path, trial: int) -> List[str]:
    return [
        str(args.python),
        str(MODEL_WORKER),
        "--output-dir",
        str(output_dir),
        "--flaggems-root",
        str(args.flaggems_root),
        "--library",
        str(args.optimized_library),
        "--precision",
        "fp32",
        "--nodes",
        "1050",
        "--layers",
        "4",
        "--warmup",
        str(args.model_warmup),
        "--repetitions",
        str(args.model_repetitions),
        "--bootstrap-resamples",
        str(args.worker_bootstrap_resamples),
        "--trial",
        str(trial),
        "--seed",
        str(args.seed),
        "--stofm-revision",
        args.stofm_revision,
        "--flaggems-revision",
        args.flaggems_revision,
    ]


def _validate_sources(result: Dict[str, Any], args: argparse.Namespace, library: Path) -> None:
    sources = result["sources"]
    if sources["stofm_revision"] != args.stofm_revision:
        raise ValueError("worker recorded an unexpected SToFM revision")
    if sources["flaggems_revision"] != args.flaggems_revision:
        raise ValueError("worker recorded an unexpected FlagGems revision")
    if sources["musa_library_sha256"] != _sha256(library):
        raise ValueError("worker recorded an unexpected MUSA library hash")


def _aggregate_operator_kind(
    trials: Sequence[Dict[str, Any]],
    *,
    candidate_kind: str,
    bootstrap_resamples: int,
) -> List[Dict[str, Any]]:
    rows: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for trial in trials:
        for row in trial["results"]:
            rows.setdefault(
                (row["operator"], row["precision"], int(row["nodes"])), []
            ).append(row)
    aggregate = []
    for (operator, precision, nodes), values in sorted(rows.items()):
        if len(values) != len(trials):
            raise ValueError(f"missing independent trial for {operator}/{precision}/{nodes}")
        reference_p50 = [row["torch_reference"]["p50_ms"] for row in values]
        candidate_p50 = [row["flagos_candidate"]["p50_ms"] for row in values]
        speedup = _paired_bootstrap_speedup(
            [
                (
                    row["torch_reference"]["samples_ms"],
                    row["flagos_candidate"]["samples_ms"],
                )
                for row in values
            ],
            resamples=bootstrap_resamples,
            seed=20260816 + nodes + len(operator) + len(precision),
        )
        aggregate.append(
            {
                "candidate_kind": candidate_kind,
                "candidate_label": values[0]["candidate_label"],
                "operator": operator,
                "operator_name": OPERATOR_NAMES[operator],
                "precision": precision,
                "nodes": nodes,
                "run_count": len(values),
                "torch_p50_ms": _summary(reference_p50),
                "flagos_p50_ms": _summary(candidate_p50),
                "speedup_over_torch": speedup,
                "latency_reduction_percent": (1.0 - 1.0 / speedup["point_estimate"]) * 100.0,
                "correctness": {
                    "oracle": values[0]["validation"]["oracle"],
                    "max_abs_error_across_trials": max(
                        row["validation"]["errors"]["max_abs_error"] for row in values
                    ),
                    "max_relative_error_across_trials": max(
                        row["validation"]["errors"]["max_relative_error"] for row in values
                    ),
                },
                "raw_device_samples": {
                    "torch": sum(
                        len(row["torch_reference"]["samples_ms"]) for row in values
                    ),
                    "flagos": sum(
                        len(row["flagos_candidate"]["samples_ms"]) for row in values
                    ),
                },
            }
        )
    return aggregate


def _aggregate_model(
    trials: Sequence[Dict[str, Any]], *, bootstrap_resamples: int
) -> List[Dict[str, Any]]:
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for trial in trials:
        for row in trial["results"]:
            rows.setdefault(row["stage"], []).append(row)
    baseline = rows["pure_pytorch"]
    aggregate = []
    for stage, values in rows.items():
        if any(row["status"] == "unavailable" for row in values):
            aggregate.append(
                {
                    "stage": stage,
                    "stage_name": STAGE_NAMES[stage],
                    "status": "unavailable",
                    "reason": values[0]["reason"],
                }
            )
            continue
        if len(values) != len(trials):
            raise ValueError(f"missing independent model trial for {stage}")
        summary: Dict[str, Any] = {
            "stage": stage,
            "stage_name": STAGE_NAMES[stage],
            "status": "measured",
            "run_count": len(values),
            "p50_ms": _summary([row["p50_ms"] for row in values]),
            "max_abs_error_across_trials": max(
                row["validation"]["errors"]["max_abs_error"] for row in values
            ),
            "raw_device_samples": sum(len(row["samples_ms"]) for row in values),
        }
        if stage != "pure_pytorch":
            speedup = _paired_bootstrap_speedup(
                [
                    (reference["samples_ms"], candidate["samples_ms"])
                    for reference, candidate in zip(baseline, values)
                ],
                resamples=bootstrap_resamples,
                seed=20260816 + len(stage),
            )
            summary["speedup_over_pure_pytorch"] = speedup
            summary["latency_reduction_percent"] = (
                1.0 - 1.0 / speedup["point_estimate"]
            ) * 100.0
        aggregate.append(summary)
    return aggregate


def _primary_comparison(operator_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in operator_rows:
        if row["nodes"] == 1050 and row["precision"] == "fp32":
            selected[(row["candidate_kind"], row["operator"])] = row
    comparisons = []
    for operator in OPERATOR_NAMES:
        initial = selected[("initial_registered_implementation", operator)]
        optimized = selected[("optimized_flagos_backend", operator)]
        initial_ms = initial["flagos_p50_ms"]["median"]
        optimized_ms = optimized["flagos_p50_ms"]["median"]
        comparisons.append(
            {
                "operator": operator,
                "operator_name": OPERATOR_NAMES[operator],
                "torch_p50_ms": optimized["torch_p50_ms"]["median"],
                "initial_flagos_p50_ms": initial_ms,
                "optimized_flagos_p50_ms": optimized_ms,
                "optimized_speedup_over_torch": optimized["speedup_over_torch"],
                "optimized_speedup_over_initial_flagos": initial_ms / optimized_ms,
                "optimized_latency_reduction_vs_initial_percent": (
                    1.0 - optimized_ms / initial_ms
                ) * 100.0,
            }
        )
    return comparisons


def _write_csv(output_dir: Path, summary: Dict[str, Any]) -> None:
    with (output_dir / "operator_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "candidate",
            "operator",
            "precision",
            "nodes",
            "runs",
            "torch_p50_ms",
            "flagos_p50_ms",
            "speedup_over_torch",
            "speedup_ci_low",
            "speedup_ci_high",
            "latency_reduction_percent",
            "max_abs_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["operator_results"]:
            ci = row["speedup_over_torch"]["bootstrap_95_ci"]
            writer.writerow(
                {
                    "candidate": row["candidate_label"],
                    "operator": row["operator_name"],
                    "precision": row["precision"],
                    "nodes": row["nodes"],
                    "runs": row["run_count"],
                    "torch_p50_ms": row["torch_p50_ms"]["median"],
                    "flagos_p50_ms": row["flagos_p50_ms"]["median"],
                    "speedup_over_torch": row["speedup_over_torch"]["point_estimate"],
                    "speedup_ci_low": ci[0],
                    "speedup_ci_high": ci[1],
                    "latency_reduction_percent": row["latency_reduction_percent"],
                    "max_abs_error": row["correctness"]["max_abs_error_across_trials"],
                }
            )


def _write_markdown(output_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# MTT S4000 MUSA Formal Optimization Evidence",
        "",
        f"Independent process trials: `{summary['trial_count']}`. All timings use MUSA device events inside `torch.inference_mode()`. ",
        "",
        "## Primary operator comparison (N=1050, FP32)",
        "",
        "| Operator | Pure PyTorch (ms) | Initial FlagOS MUSA (ms) | Optimized FlagOS MUSA (ms) | Optimized vs PyTorch | Optimized vs initial |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["primary_operator_comparison"]:
        lines.append(
            f"| {row['operator_name']} | {row['torch_p50_ms']:.4f} | "
            f"{row['initial_flagos_p50_ms']:.4f} | {row['optimized_flagos_p50_ms']:.4f} | "
            f"{row['optimized_speedup_over_torch']['point_estimate']:.3f}x | "
            f"{row['optimized_speedup_over_initial_flagos']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## End-to-end SToFM (N=1050, four layers, FP32)",
            "",
            "| Execution path | p50 (ms) | Speedup over pure PyTorch | Max absolute error |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["model_results"]:
        if row["status"] == "unavailable":
            lines.append(f"| {row['stage_name']} | unavailable | unavailable | unavailable |")
            continue
        speedup = row.get("speedup_over_pure_pytorch")
        speedup_text = f"{speedup['point_estimate']:.3f}x" if speedup else "1.000x"
        lines.append(
            f"| {row['stage_name']} | {row['p50_ms']['median']:.4f} | {speedup_text} | "
            f"{row['max_abs_error_across_trials']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The frozen upstream FlagOS availability probe is preserved separately. An unavailable backend is never replaced with a different implementation or a fabricated timing.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--flaggems-root", type=Path, required=True)
    parser.add_argument("--stock-flaggems-root", type=Path, required=True)
    parser.add_argument("--initial-library", type=Path, required=True)
    parser.add_argument("--optimized-library", type=Path, required=True)
    parser.add_argument("--stofm-revision", required=True)
    parser.add_argument("--flaggems-revision", required=True)
    parser.add_argument(
        "--stock-flaggems-revision",
        default="03bf364ede763d573d5c30124d554283a209ab85",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--matrix-shapes", default="256,512,1050,2048")
    parser.add_argument("--matrix-precisions", default="fp32,fp16,bf16")
    parser.add_argument("--matrix-warmup", type=int, default=5)
    parser.add_argument("--matrix-repetitions", type=int, default=10)
    parser.add_argument("--initial-warmup", type=int, default=5)
    parser.add_argument("--initial-repetitions", type=int, default=10)
    parser.add_argument("--model-warmup", type=int, default=5)
    parser.add_argument("--model-repetitions", type=int, default=15)
    parser.add_argument("--worker-bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--aggregate-bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    if args.trials < 5:
        raise ValueError("formal MTT S4000 reporting requires at least five independent trials")
    for path in (
        args.python,
        args.initial_library,
        args.optimized_library,
        args.flaggems_root,
        args.stock_flaggems_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stock_dir = args.output_dir / "frozen-upstream-flagos-probe"
    _run(
        [
            str(args.python),
            str(STOCK_PROBE),
            "--stock-root",
            str(args.stock_flaggems_root),
            "--output-dir",
            str(stock_dir),
            "--expected-revision",
            args.stock_flaggems_revision,
        ],
        log_path=stock_dir / "runner.log",
        accepted_codes=(0, 2),
    )
    stock_probe = _read_json(stock_dir / "result.json")

    initial_trials = []
    optimized_trials = []
    model_trials = []
    task_orders = (
        ("initial", "optimized", "model"),
        ("optimized", "model", "initial"),
        ("model", "initial", "optimized"),
    )
    for trial in range(1, args.trials + 1):
        trial_dir = args.output_dir / f"trial-{trial:02d}"
        tasks = {
            "initial": (
                _matrix_command(
                    args,
                    output_dir=trial_dir / "initial-flagos-operators",
                    trial=trial,
                    candidate="direct-privateuse1",
                    library=args.initial_library,
                    shapes="1050",
                    precisions="fp32",
                    warmup=args.initial_warmup,
                    repetitions=args.initial_repetitions,
                ),
                trial_dir / "initial-flagos-operators" / "worker.log",
            ),
            "optimized": (
                _matrix_command(
                    args,
                    output_dir=trial_dir / "optimized-operator-matrix",
                    trial=trial,
                    candidate="flagos-backend",
                    library=args.optimized_library,
                    shapes=args.matrix_shapes,
                    precisions=args.matrix_precisions,
                    warmup=args.matrix_warmup,
                    repetitions=args.matrix_repetitions,
                ),
                trial_dir / "optimized-operator-matrix" / "worker.log",
            ),
            "model": (
                _model_command(
                    args, output_dir=trial_dir / "end-to-end-model", trial=trial
                ),
                trial_dir / "end-to-end-model" / "worker.log",
            ),
        }
        for task_name in task_orders[(trial - 1) % len(task_orders)]:
            command, log_path = tasks[task_name]
            _run(command, log_path=log_path)

        initial = _read_json(trial_dir / "initial-flagos-operators" / "result.json")
        optimized = _read_json(trial_dir / "optimized-operator-matrix" / "result.json")
        model = _read_json(trial_dir / "end-to-end-model" / "result.json")
        _validate_sources(initial, args, args.initial_library)
        _validate_sources(optimized, args, args.optimized_library)
        _validate_sources(model, args, args.optimized_library)
        initial_trials.append(initial)
        optimized_trials.append(optimized)
        model_trials.append(model)

    operator_results = _aggregate_operator_kind(
        initial_trials,
        candidate_kind="initial_registered_implementation",
        bootstrap_resamples=args.aggregate_bootstrap_resamples,
    )
    operator_results.extend(
        _aggregate_operator_kind(
            optimized_trials,
            candidate_kind="optimized_flagos_backend",
            bootstrap_resamples=args.aggregate_bootstrap_resamples,
        )
    )
    summary = {
        "schema_version": 1,
        "trial_count": args.trials,
        "source_revisions": {
            "stofm": args.stofm_revision,
            "flaggems": args.flaggems_revision,
            "frozen_upstream_flaggems": args.stock_flaggems_revision,
        },
        "library_sha256": {
            "initial_registered_implementation": _sha256(args.initial_library),
            "optimized_flagos_backend": _sha256(args.optimized_library),
        },
        "protocol": {
            "independent_processes": True,
            "trial_order": "three-position cyclic rotation",
            "timer": "MUSA device events with per-sample synchronization",
            "inference_mode": True,
            "compile_included": False,
            "aggregate_bootstrap_resamples": args.aggregate_bootstrap_resamples,
        },
        "frozen_upstream_flagos": stock_probe,
        "operator_results": operator_results,
        "primary_operator_comparison": _primary_comparison(operator_results),
        "model_results": _aggregate_model(
            model_trials, bootstrap_resamples=args.aggregate_bootstrap_resamples
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir, summary)
    _write_markdown(args.output_dir, summary)
    manifest = {
        "summary_sha256": _sha256(args.output_dir / "summary.json"),
        "operator_summary_sha256": _sha256(args.output_dir / "operator_summary.csv"),
        "report_sha256": _sha256(args.output_dir / "report.md"),
        "trial_result_files": sorted(
            str(path.relative_to(args.output_dir))
            for path in args.output_dir.glob("trial-*/**/result.json")
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
