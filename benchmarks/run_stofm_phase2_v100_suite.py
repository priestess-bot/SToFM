#!/usr/bin/env python3
"""Orchestrate and verify the SToFM PHASE 2 V100 training benchmark suite."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "benchmarks" / "stofm_phase2_v100_worker.py"
FLAGGEMS_ROOT = ROOT.parent / "FlagGems-stofm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.stofm_phase2_v100_worker import ROUTES


PRIMARY_PROFILE_ROUTES = {
    "torch_fused",
    "flagos_reference_scalar",
    "flagos_native_scalar",
}


def _quantile(values: Sequence[float], percentile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile))


def _flat_stats(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "std_ms": float(statistics.pstdev(values)),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _hierarchical_speedup(
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("hierarchical bootstrap requires paired non-empty trial blocks")
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples, dtype=np.float64)
    trial_count = len(baseline)
    for index in range(resamples):
        sampled_baseline: List[float] = []
        sampled_candidate: List[float] = []
        for trial_index in rng.integers(0, trial_count, size=trial_count):
            baseline_block = np.asarray(baseline[trial_index], dtype=np.float64)
            candidate_block = np.asarray(candidate[trial_index], dtype=np.float64)
            sampled_baseline.extend(
                rng.choice(baseline_block, size=baseline_block.size, replace=True).tolist()
            )
            sampled_candidate.extend(
                rng.choice(candidate_block, size=candidate_block.size, replace=True).tolist()
            )
        ratios[index] = np.median(sampled_baseline) / np.median(sampled_candidate)
    baseline_flat = [value for block in baseline for value in block]
    candidate_flat = [value for block in candidate for value in block]
    return {
        "speedup": float(statistics.median(baseline_flat) / statistics.median(candidate_flat)),
        "latency_reduction_percent": float(
            (1.0 - statistics.median(candidate_flat) / statistics.median(baseline_flat))
            * 100.0
        ),
        "bootstrap_95_ci": {
            "lower": float(np.quantile(ratios, 0.025)),
            "upper": float(np.quantile(ratios, 0.975)),
        },
        "method": "paired hierarchical bootstrap over trial blocks and independent CUDA-event samples",
        "resamples": resamples,
    }


def _metric_blocks(
    trials: Sequence[Mapping[str, Any]], route_id: str, metric: str
) -> List[List[float]]:
    return [
        list(trial[route_id]["timing"]["metrics"][metric]["samples_ms"])
        for trial in trials
    ]


def _aggregate(
    trials: Sequence[Mapping[str, Any]], *, routes: Sequence[str], resamples: int
) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {"routes": {}, "comparisons": {}}
    metrics = ("forward_ms", "backward_ms", "optimizer_ms", "step_ms")
    for route_id in routes:
        route_summary: Dict[str, Any] = {
            "display_name": ROUTES[route_id]["display_name"],
            "metrics": {},
        }
        for metric in metrics:
            blocks = _metric_blocks(trials, route_id, metric)
            flat = [value for block in blocks for value in block]
            trial_medians = [float(statistics.median(block)) for block in blocks]
            mean_trial_median = statistics.fmean(trial_medians)
            route_summary["metrics"][metric] = {
                **_flat_stats(flat),
                "raw_sample_count": len(flat),
                "trial_medians_ms": trial_medians,
                "median_of_trial_medians_ms": float(statistics.median(trial_medians)),
                "trial_median_min_ms": float(min(trial_medians)),
                "trial_median_max_ms": float(max(trial_medians)),
                "trial_median_cv_percent": float(
                    statistics.pstdev(trial_medians) / mean_trial_median * 100.0
                )
                if mean_trial_median
                else 0.0,
            }
        memory_rows = [trial[route_id]["timing"]["memory"] for trial in trials]
        route_summary["memory"] = {
            key: int(max(row[key] for row in memory_rows)) for key in memory_rows[0]
        }
        route_summary["throughput"] = {
            key: float(statistics.median([trial[route_id]["timing"]["throughput"][key] for trial in trials]))
            for key in trials[0][route_id]["timing"]["throughput"]
        }
        aggregate["routes"][route_id] = route_summary

    comparisons = (
        ("optimized_flagos_vs_torch_fused", "torch_fused", "flagos_native_scalar"),
        (
            "optimized_flagos_vs_initial_flagos",
            "flagos_reference_scalar",
            "flagos_native_scalar",
        ),
        (
            "native_operators_only",
            "flagos_reference_scalar",
            "flagos_native_scalar",
        ),
        (
            "fused_optimizer_only",
            "flagos_reference_scalar",
            "flagos_reference_fused",
        ),
        (
            "fused_optimizer_on_native_route",
            "flagos_native_scalar",
            "flagos_native_fused",
        ),
        ("torch_fused_vs_torch_scalar", "torch_scalar", "torch_fused"),
    )
    for comparison_index, (name, baseline, candidate) in enumerate(comparisons):
        if baseline not in routes or candidate not in routes:
            continue
        aggregate["comparisons"][name] = {
            "baseline": baseline,
            "candidate": candidate,
            "metrics": {
                metric: _hierarchical_speedup(
                    _metric_blocks(trials, baseline, metric),
                    _metric_blocks(trials, candidate, metric),
                    resamples=resamples,
                    seed=20260830 + comparison_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(metrics)
            },
        }
    return aggregate


def _tensor_error(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    actual = actual.float()
    reference = reference.float()
    difference = (actual - reference).abs()
    relative = difference / reference.abs().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": float(relative.max()) if relative.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
    }


def _mapping_errors(
    actual: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]
) -> Dict[str, Any]:
    if set(actual) != set(reference):
        raise AssertionError("correctness snapshot tensor names differ")
    rows = {name: _tensor_error(actual[name], reference[name]) for name in reference}
    return {
        "tensor_count": len(rows),
        "max_abs": max((row["max_abs"] for row in rows.values()), default=0.0),
        "max_rel": max((row["max_rel"] for row in rows.values()), default=0.0),
        "mean_of_tensor_mean_abs": float(
            statistics.fmean(row["mean_abs"] for row in rows.values())
        )
        if rows
        else 0.0,
        "worst_abs_tensor": max(rows, key=lambda name: rows[name]["max_abs"]) if rows else None,
        "per_tensor": rows,
    }


def _flatten_optimizer_state(
    value: Mapping[str, Mapping[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    return {
        f"{name}.{state_name}": tensor
        for name, state in value.items()
        for state_name, tensor in state.items()
    }


def _verify_correctness(
    output: Path,
    trials: Sequence[Mapping[str, Any]],
    routes: Sequence[str],
) -> Dict[str, Any]:
    reference_route = "torch_scalar"
    if reference_route not in routes:
        raise ValueError("correctness verification requires the torch_scalar route")
    reference_path = output / "trial-01" / reference_route / "first_step.pt"
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    rows: Dict[str, Any] = {}
    thresholds = {
        "loss_max_abs": 2e-5,
        "gradient_max_abs": 2e-4,
        "parameter_max_abs": 2e-5,
        "optimizer_state_max_abs": 2e-5,
    }
    for route_id in routes:
        route_path = output / "trial-01" / route_id / "first_step.pt"
        actual = torch.load(route_path, map_location="cpu", weights_only=False)
        loss_errors = {
            name: abs(float(actual["losses"][name]) - float(reference["losses"][name]))
            for name in reference["losses"]
        }
        gradient_errors = _mapping_errors(actual["gradients"], reference["gradients"])
        parameter_errors = _mapping_errors(
            actual["updated_parameters"], reference["updated_parameters"]
        )
        optimizer_errors = _mapping_errors(
            _flatten_optimizer_state(actual["optimizer_state"]),
            _flatten_optimizer_state(reference["optimizer_state"]),
        )
        passed = (
            max(loss_errors.values()) <= thresholds["loss_max_abs"]
            and gradient_errors["max_abs"] <= thresholds["gradient_max_abs"]
            and parameter_errors["max_abs"] <= thresholds["parameter_max_abs"]
            and optimizer_errors["max_abs"] <= thresholds["optimizer_state_max_abs"]
        )
        rows[route_id] = {
            "display_name": ROUTES[route_id]["display_name"],
            "loss_abs_errors": loss_errors,
            "gradient_errors": gradient_errors,
            "updated_parameter_errors": parameter_errors,
            "optimizer_state_errors": optimizer_errors,
            "passed": passed,
        }
    hashes_consistent = True
    for route_id in routes:
        initial_hashes = {
            trial[route_id]["correctness"]["initial_parameter_sha256"] for trial in trials
        }
        batch_hashes = {trial[route_id]["batch_sha256"] for trial in trials}
        hashes_consistent &= len(initial_hashes) == 1 and len(batch_hashes) == 1
    result = {
        "reference_route": reference_route,
        "thresholds": thresholds,
        "hashes_consistent_across_trials": hashes_consistent,
        "routes": rows,
        "passed": hashes_consistent and all(row["passed"] for row in rows.values()),
    }
    (output / "correctness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _write_summary(result: Mapping[str, Any], output: Path) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# SToFM PHASE 2 V100 训练性能结果",
        "",
        "## 三路主结论",
        "",
        "| 路线 | 完整训练步 median | forward | backward | optimizer | 峰值显存 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route_id in result["routes"]:
        row = aggregate["routes"][route_id]
        metrics = row["metrics"]
        lines.append(
            f"| {row['display_name']} | {metrics['step_ms']['median_ms']:.4f} ms | "
            f"{metrics['forward_ms']['median_ms']:.4f} ms | "
            f"{metrics['backward_ms']['median_ms']:.4f} ms | "
            f"{metrics['optimizer_ms']['median_ms']:.4f} ms | "
            f"{row['memory']['peak_allocated_bytes'] / 2**20:.1f} MiB |"
        )
    lines.extend(["", "## 优化归因", ""])
    for name, comparison in aggregate["comparisons"].items():
        speedup = comparison["metrics"]["step_ms"]
        interval = speedup["bootstrap_95_ci"]
        lines.append(
            f"- `{name}`：{speedup['speedup']:.3f}x，95% bootstrap CI "
            f"[{interval['lower']:.3f}x, {interval['upper']:.3f}x]。"
        )
    lines.extend(
        [
            "",
            "## 正确性",
            "",
            f"严格第一步对照：{'通过' if result['correctness']['passed'] else '失败'}。",
            "",
            "所有原始 CUDA event 样本、第一步梯度/参数状态、profile trace、运行命令和环境信息均保存在本目录。",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(output: Path) -> Dict[str, str]:
    checksums = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.json":
            continue
        checksums[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "SHA256SUMS.json").write_text(
        json.dumps(checksums, indent=2), encoding="utf-8"
    )
    return checksums


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=list(ROUTES))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=384)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.trials < 2 or args.warmup < 1 or args.repetitions < 1:
        raise ValueError("trials must be at least two; warmup and repetitions must be positive")
    if "torch_scalar" not in args.routes:
        raise ValueError("the suite requires torch_scalar for strict correctness comparison")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command_records = []
    trials: List[Dict[str, Any]] = []
    common = [
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--batch-size",
        str(args.batch_size),
        "--nodes",
        str(args.nodes),
        "--input-dim",
        str(args.input_dim),
        "--embedding-dim",
        str(args.embedding_dim),
        "--heads",
        str(args.heads),
        "--gaussian-hidden-dim",
        str(args.gaussian_hidden_dim),
        "--layers",
        str(args.layers),
        "--seed",
        str(args.seed),
    ]
    for trial_number in range(1, args.trials + 1):
        trial: Dict[str, Any] = {}
        shift = (trial_number - 1) % len(args.routes)
        order = args.routes[shift:] + args.routes[:shift]
        for route_id in order:
            route_output = output / f"trial-{trial_number:02d}" / route_id
            command = [
                sys.executable,
                str(WORKER),
                "--route",
                route_id,
                "--trial",
                str(trial_number),
                "--output",
                str(route_output),
                *common,
            ]
            if args.profile and trial_number == 1 and route_id in PRIMARY_PROFILE_ROUTES:
                command.append("--profile")
            else:
                command.append("--no-profile")
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            record = {
                "trial": trial_number,
                "route": route_id,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            command_records.append(record)
            (output / "execution_log.json").write_text(
                json.dumps(command_records, indent=2), encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed for trial {trial_number}, route {route_id}; "
                    "see execution_log.json"
                )
            trial[route_id] = json.loads(
                (route_output / "result.json").read_text(encoding="utf-8")
            )
        trials.append(trial)

    environments = {
        json.dumps(trial[route_id]["environment"], sort_keys=True)
        for trial in trials
        for route_id in args.routes
    }
    revisions = {
        json.dumps(trial[route_id]["revisions"], sort_keys=True)
        for trial in trials
        for route_id in args.routes
    }
    workloads = {
        json.dumps(trial[route_id]["workload"], sort_keys=True)
        for trial in trials
        for route_id in args.routes
    }
    if len(environments) != 1 or len(revisions) != 1 or len(workloads) != 1:
        raise AssertionError("environment, revision, or workload drift detected")
    environment = trials[0][args.routes[0]]["environment"]
    if "V100" not in environment["device"]:
        raise AssertionError(f"expected V100, observed {environment['device']}")
    correctness = _verify_correctness(output, trials, args.routes)
    if not correctness["passed"]:
        raise AssertionError("strict first-step correctness comparison failed")
    aggregate = _aggregate(
        trials, routes=args.routes, resamples=args.bootstrap_resamples
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "benchmark": "SToFM PHASE 2 V100 FP32 training",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "VERIFIED",
            "version_label": "stofm_phase2_v100_training_suite_v1",
            "upstream_dependencies": ["deps/flagos-training.lock.json"],
        },
        "routes": list(args.routes),
        "trial_count": args.trials,
        "raw_samples_per_route": args.trials * args.repetitions,
        "protocol": {
            "independent_process_per_route_and_trial": True,
            "rotating_route_order": True,
            "warmup": args.warmup,
            "repetitions_per_trial": args.repetitions,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "environment": environment,
        "revisions": trials[0][args.routes[0]]["revisions"],
        "workload": trials[0][args.routes[0]]["workload"],
        "correctness": correctness,
        "aggregate": aggregate,
        "profile_routes": sorted(
            PRIMARY_PROFILE_ROUTES.intersection(args.routes) if args.profile else set()
        ),
    }
    (output / "suite.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_summary(result, output)
    checksums = _write_checksums(output)
    result["evidence_file_count"] = len(checksums)
    (output / "suite.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_checksums(output)
    print(json.dumps({"status": "passed", "output": str(output)}))


if __name__ == "__main__":
    main()
