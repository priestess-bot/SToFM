#!/usr/bin/env python3
"""Run and aggregate the PHASE 2 Gaussian/Pair V100 training matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "benchmarks" / "stofm_phase2_v100_operator_worker.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_stofm_phase2_v100_suite import (
    _flat_stats,
    _hierarchical_speedup,
    _write_checksums,
)
from benchmarks.stofm_phase2_v100_operator_worker import IMPLEMENTATIONS


OPERATORS = ("gaussian", "pair")


def _blocks(
    trials: Sequence[Mapping[str, Any]],
    operator: str,
    implementation: str,
    metric: str,
) -> List[List[float]]:
    return [
        list(trial[operator][implementation]["metrics"][metric]["samples_ms"])
        for trial in trials
    ]


def _aggregate(
    trials: Sequence[Mapping[str, Any]], *, resamples: int
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for operator_index, operator in enumerate(OPERATORS):
        operator_result: Dict[str, Any] = {"implementations": {}, "comparisons": {}}
        for implementation in IMPLEMENTATIONS:
            metrics = {}
            for metric in ("forward_ms", "backward_ms", "total_ms"):
                blocks = _blocks(trials, operator, implementation, metric)
                flat = [sample for block in blocks for sample in block]
                medians = [float(statistics.median(block)) for block in blocks]
                metrics[metric] = {
                    **_flat_stats(flat),
                    "raw_sample_count": len(flat),
                    "trial_medians_ms": medians,
                    "median_of_trial_medians_ms": float(statistics.median(medians)),
                }
            operator_result["implementations"][implementation] = {
                "display_name": IMPLEMENTATIONS[implementation],
                "metrics": metrics,
                "peak_allocated_bytes": int(
                    max(
                        trial[operator][implementation]["memory"]["peak_allocated_bytes"]
                        for trial in trials
                    )
                ),
                "correctness": {
                    "max_output_abs": max(
                        trial[operator][implementation]["correctness"]["outputs"]["max_abs"]
                        for trial in trials
                    ),
                    "max_gradient_abs": max(
                        trial[operator][implementation]["correctness"]["gradients"]["max_abs"]
                        for trial in trials
                    ),
                },
            }
        for comparison_index, (name, baseline, candidate) in enumerate(
            (
                ("native_vs_torch", "torch", "flagos_native"),
                ("native_vs_initial_flagos", "flagos_reference", "flagos_native"),
                ("initial_flagos_vs_torch", "torch", "flagos_reference"),
            )
        ):
            operator_result["comparisons"][name] = {
                "baseline": baseline,
                "candidate": candidate,
                "metrics": {
                    metric: _hierarchical_speedup(
                        _blocks(trials, operator, baseline, metric),
                        _blocks(trials, operator, candidate, metric),
                        resamples=resamples,
                        seed=(
                            20260830
                            + operator_index * 1000
                            + comparison_index * 100
                            + metric_index
                        ),
                    )
                    for metric_index, metric in enumerate(
                        ("forward_ms", "backward_ms", "total_ms")
                    )
                },
            }
        result[operator] = operator_result
    return result


def _summary(result: Mapping[str, Any], output: Path) -> None:
    labels = {"gaussian": "Gaussian pair bias", "pair": "Pair-score attention"}
    lines = [
        "# SToFM PHASE 2 V100 逐算子训练性能",
        "",
        "| 算子 | 实现 | forward | backward | 总计 | 峰值显存 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for operator in OPERATORS:
        for implementation, row in result["aggregate"][operator]["implementations"].items():
            metrics = row["metrics"]
            lines.append(
                f"| {labels[operator]} | {row['display_name']} | "
                f"{metrics['forward_ms']['median_ms']:.4f} ms | "
                f"{metrics['backward_ms']['median_ms']:.4f} ms | "
                f"{metrics['total_ms']['median_ms']:.4f} ms | "
                f"{row['peak_allocated_bytes'] / 2**20:.1f} MiB |"
            )
    lines.extend(["", "## 原生实现相对初始 FlagOS", ""])
    for operator in OPERATORS:
        speedup = result["aggregate"][operator]["comparisons"][
            "native_vs_initial_flagos"
        ]["metrics"]["total_ms"]
        interval = speedup["bootstrap_95_ci"]
        lines.append(
            f"- {labels[operator]}：{speedup['speedup']:.3f}x，95% CI "
            f"[{interval['lower']:.3f}x, {interval['upper']:.3f}x]。"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.trials < 2 or args.warmup < 1 or args.repetitions < 1:
        raise ValueError("trials must be at least two; warmup and repetitions must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command_records = []
    trials: List[Dict[str, Any]] = []
    for trial_number in range(1, args.trials + 1):
        trial: Dict[str, Any] = {operator: {} for operator in OPERATORS}
        matrix = [
            (operator, implementation)
            for operator in OPERATORS
            for implementation in IMPLEMENTATIONS
        ]
        shift = (trial_number - 1) % len(matrix)
        for operator, implementation in matrix[shift:] + matrix[:shift]:
            route_output = (
                output
                / f"trial-{trial_number:02d}"
                / operator
                / implementation
            )
            command = [
                sys.executable,
                str(WORKER),
                "--operator",
                operator,
                "--implementation",
                implementation,
                "--trial",
                str(trial_number),
                "--output",
                str(route_output),
                "--warmup",
                str(args.warmup),
                "--repetitions",
                str(args.repetitions),
                "--batch-size",
                str(args.batch_size),
                "--nodes",
                str(args.nodes),
                "--embedding-dim",
                str(args.embedding_dim),
                "--heads",
                str(args.heads),
                "--gaussian-hidden-dim",
                str(args.gaussian_hidden_dim),
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            command_records.append(
                {
                    "trial": trial_number,
                    "operator": operator,
                    "implementation": implementation,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            (output / "execution_log.json").write_text(
                json.dumps(command_records, indent=2), encoding="utf-8"
            )
            if completed.returncode:
                raise RuntimeError(
                    f"operator worker failed: trial={trial_number}, "
                    f"operator={operator}, implementation={implementation}"
                )
            trial[operator][implementation] = json.loads(
                (route_output / "result.json").read_text(encoding="utf-8")
            )
        trials.append(trial)

    revisions = {
        json.dumps(trial[operator][implementation]["revisions"], sort_keys=True)
        for trial in trials
        for operator in OPERATORS
        for implementation in IMPLEMENTATIONS
    }
    environments = {
        json.dumps(trial[operator][implementation]["environment"], sort_keys=True)
        for trial in trials
        for operator in OPERATORS
        for implementation in IMPLEMENTATIONS
    }
    if len(revisions) != 1 or len(environments) != 1:
        raise AssertionError("operator suite revision or environment drift detected")
    if not all(
        trial[operator][implementation]["correctness"]["passed"]
        for trial in trials
        for operator in OPERATORS
        for implementation in IMPLEMENTATIONS
    ):
        raise AssertionError("operator correctness gate failed")
    aggregate = _aggregate(trials, resamples=args.bootstrap_resamples)
    result = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "SToFM PHASE 2 V100 operator training matrix",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "VERIFIED",
            "version_label": "stofm_phase2_v100_operator_suite_v1",
        },
        "trial_count": args.trials,
        "raw_samples_per_implementation": args.trials * args.repetitions,
        "protocol": {
            "independent_process_per_implementation_and_trial": True,
            "rotating_execution_order": True,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "revisions": trials[0]["gaussian"]["torch"]["revisions"],
        "environment": trials[0]["gaussian"]["torch"]["environment"],
        "workload": trials[0]["gaussian"]["torch"]["workload"],
        "aggregate": aggregate,
    }
    (output / "suite.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _summary(result, output)
    checksums = _write_checksums(output)
    result["evidence_file_count"] = len(checksums)
    (output / "suite.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_checksums(output)
    print(json.dumps({"status": "passed", "output": str(output)}))


if __name__ == "__main__":
    main()
