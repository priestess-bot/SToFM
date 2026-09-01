#!/usr/bin/env python3
"""Validate the two-shape SToFM Stage A V100 acceptance contract."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


BASELINE_ROUTE = "torch_fused"
CANDIDATE_ROUTE = "flagos_vendor_native_fused_v100_tuned"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_shape(root: Path) -> Tuple[Dict[str, Any], List[List[float]], List[List[float]]]:
    suite_path = root / "suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    baseline_blocks = []
    candidate_blocks = []
    for trial in range(1, int(suite["trial_count"]) + 1):
        trial_root = root / f"trial-{trial:02d}"
        baseline = json.loads(
            (trial_root / BASELINE_ROUTE / "result.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (trial_root / CANDIDATE_ROUTE / "result.json").read_text(encoding="utf-8")
        )
        baseline_blocks.append(
            [float(value) for value in baseline["timing"]["metrics"]["step_ms"]["samples_ms"]]
        )
        candidate_blocks.append(
            [float(value) for value in candidate["timing"]["metrics"]["step_ms"]["samples_ms"]]
        )
    return suite, baseline_blocks, candidate_blocks


def _resampled_speedup(
    rng: np.random.Generator,
    baseline: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> float:
    sampled_baseline: List[float] = []
    sampled_candidate: List[float] = []
    trial_count = len(baseline)
    for trial_index in rng.integers(0, trial_count, size=trial_count):
        baseline_block = np.asarray(baseline[trial_index], dtype=np.float64)
        candidate_block = np.asarray(candidate[trial_index], dtype=np.float64)
        sampled_baseline.extend(
            rng.choice(baseline_block, size=baseline_block.size, replace=True).tolist()
        )
        sampled_candidate.extend(
            rng.choice(candidate_block, size=candidate_block.size, replace=True).tolist()
        )
    return float(statistics.median(sampled_baseline) / statistics.median(sampled_candidate))


def _profile_provenance(root: Path) -> Mapping[str, Any]:
    result = json.loads(
        (root / "trial-01" / CANDIDATE_ROUTE / "result.json").read_text(
            encoding="utf-8"
        )
    )
    return result.get("profile", {}).get("gemm_provenance", {})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shapes = {
        "representative": args.representative.resolve(),
        "production": args.production.resolve(),
    }
    loaded = {name: _load_shape(root) for name, root in shapes.items()}
    revisions = {
        json.dumps(suite["revisions"], sort_keys=True)
        for suite, _, _ in loaded.values()
    }
    if len(revisions) != 1:
        raise AssertionError("shape suites were not measured at identical revisions")

    shape_rows: Dict[str, Any] = {}
    for name, root in shapes.items():
        suite, baseline_blocks, candidate_blocks = loaded[name]
        baseline_flat = [value for block in baseline_blocks for value in block]
        candidate_flat = [value for block in candidate_blocks for value in block]
        baseline_summary = suite["aggregate"]["routes"][BASELINE_ROUTE]
        candidate_summary = suite["aggregate"]["routes"][CANDIDATE_ROUTE]
        provenance = _profile_provenance(root)
        shape_rows[name] = {
            "workload": suite["workload"],
            "trial_count": suite["trial_count"],
            "sample_count_per_route": len(baseline_flat),
            "torch_fused_step_median_ms": float(statistics.median(baseline_flat)),
            "vendor_tuned_step_median_ms": float(statistics.median(candidate_flat)),
            "speedup": float(statistics.median(baseline_flat) / statistics.median(candidate_flat)),
            "torch_fused_step_p95_ms": float(np.quantile(baseline_flat, 0.95)),
            "vendor_tuned_step_p95_ms": float(np.quantile(candidate_flat, 0.95)),
            "peak_memory_ratio": float(
                candidate_summary["memory"]["peak_allocated_bytes"]
                / baseline_summary["memory"]["peak_allocated_bytes"]
            ),
            "correctness_passed": bool(suite["correctness"]["passed"]),
            "native_torch_gemm_absent": bool(provenance.get("native_gemm_absent")),
            "all_profiled_gemm_owned_by_vendor_cpp": bool(
                provenance.get("all_profiled_aten_gemm_owned_by_vendor_cpp")
            ),
            "suite_sha256": _sha256(root / "suite.json"),
        }

    point_speedup = math.exp(
        statistics.fmean(math.log(row["speedup"]) for row in shape_rows.values())
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_resamples, dtype=np.float64)
    for index in range(args.bootstrap_resamples):
        per_shape = [
            _resampled_speedup(rng, loaded[name][1], loaded[name][2])
            for name in shapes
        ]
        bootstrap[index] = math.exp(statistics.fmean(math.log(value) for value in per_shape))
    interval = {
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
    }
    gates = {
        "aggregate_speedup_at_least_1_05": point_speedup >= 1.05,
        "bootstrap_lower_above_1": interval["lower"] > 1.0,
        "no_shape_slower_than_torch": min(row["speedup"] for row in shape_rows.values()) >= 1.0,
        "peak_memory_at_most_1_25x_torch": max(
            row["peak_memory_ratio"] for row in shape_rows.values()
        )
        <= 1.25,
        "strict_correctness": all(row["correctness_passed"] for row in shape_rows.values()),
        "no_native_torch_gemm": all(
            row["native_torch_gemm_absent"]
            and row["all_profiled_gemm_owned_by_vendor_cpp"]
            for row in shape_rows.values()
        ),
        "five_trials_fifty_samples": all(
            row["trial_count"] >= 5 and row["sample_count_per_route"] >= 250
            for row in shape_rows.values()
        ),
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "benchmark": "SToFM Stage A V100 two-shape acceptance",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validation",
            "verification_status": "VERIFIED" if all(gates.values()) else "FAILED",
            "version_label": "stofm_stageA_v100_acceptance_v1",
        },
        "routes": {"baseline": BASELINE_ROUTE, "candidate": CANDIDATE_ROUTE},
        "revisions": json.loads(next(iter(revisions))),
        "shapes": shape_rows,
        "aggregate": {
            "method": "equal-weight geometric mean of per-shape median speedups",
            "speedup": point_speedup,
            "latency_reduction_percent": (1.0 - 1.0 / point_speedup) * 100.0,
            "bootstrap_95_ci": interval,
            "bootstrap_method": (
                "fixed workload matrix; paired hierarchical resampling of trial blocks "
                "and CUDA-event samples within each shape"
            ),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# SToFM Stage A V100 验收",
        "",
        f"状态：**{result['status']}**",
        "",
        "| 形状 | Torch fused | FlagOS Vendor tuned | 加速 | P95 (Torch → FlagOS) | 显存比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in shape_rows.items():
        lines.append(
            f"| {name} | {row['torch_fused_step_median_ms']:.4f} ms | "
            f"{row['vendor_tuned_step_median_ms']:.4f} ms | {row['speedup']:.4f}x | "
            f"{row['torch_fused_step_p95_ms']:.4f} → {row['vendor_tuned_step_p95_ms']:.4f} ms | "
            f"{row['peak_memory_ratio']:.3f}x |"
        )
    lines.extend(
        [
            "",
            f"联合加速：**{point_speedup:.4f}x**，95% bootstrap CI "
            f"[{interval['lower']:.4f}x, {interval['upper']:.4f}x]。",
            "",
            "## 门槛",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}" for name, passed in gates.items()
    )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
