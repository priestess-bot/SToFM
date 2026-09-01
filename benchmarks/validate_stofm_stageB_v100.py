#!/usr/bin/env python3
"""Validate the two-shape SToFM Stage B self-hosted V100 contract."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


BASELINE_ROUTE = "torch_fused"
CANDIDATE_ROUTE = "flagos_self_hosted_native_fused_v100_tuned"
FORBIDDEN_DEPENDENCIES = ("cublas", "cutlass")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_shape(root: Path) -> Tuple[Dict[str, Any], List[List[float]], List[List[float]]]:
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    baseline_blocks: List[List[float]] = []
    candidate_blocks: List[List[float]] = []
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


def _candidate_result(root: Path) -> Dict[str, Any]:
    return json.loads(
        (root / "trial-01" / CANDIDATE_ROUTE / "result.json").read_text(
            encoding="utf-8"
        )
    )


def _dependency_audit(library: Path, flaggems_root: Path) -> Dict[str, Any]:
    if not library.is_file():
        raise FileNotFoundError(f"self-hosted library is missing: {library}")
    dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    symbols = subprocess.check_output(["nm", "-D", str(library)], text=True)
    source_paths = (
        flaggems_root / "cpp/lib/stofm_self_hosted_gemm.cu",
        flaggems_root / "cpp/include/flag_gems/stofm_self_hosted_gemm.h",
        flaggems_root / "src/flag_gems/experimental_ops/self_hosted_gemm.py",
        flaggems_root / "tools/build_stofm_self_hosted_gemm.py",
    )
    source_hits: Dict[str, List[str]] = {}
    for path in source_paths:
        lowered = path.read_text(encoding="utf-8").lower()
        hits = [token for token in FORBIDDEN_DEPENDENCIES if token in lowered]
        if hits:
            source_hits[str(path.relative_to(flaggems_root))] = hits
    needed_lines = [line.strip() for line in dynamic.splitlines() if "NEEDED" in line]
    dynamic_hits = [token for token in FORBIDDEN_DEPENDENCIES if token in dynamic.lower()]
    symbol_hits = [token for token in FORBIDDEN_DEPENDENCIES if token in symbols.lower()]
    return {
        "library_sha256": _sha256(library),
        "library_size_bytes": library.stat().st_size,
        "needed_entries": needed_lines,
        "source_hits": source_hits,
        "dynamic_dependency_hits": dynamic_hits,
        "dynamic_symbol_hits": symbol_hits,
        "passed": not source_hits and not dynamic_hits and not symbol_hits,
        "scope_note": (
            "The audit covers the self-hosted extension and profiled GEMM calls. "
            "The PyTorch binary distribution may preload external BLAS libraries "
            "independently; Stage B does not link or call them."
        ),
    }


def _profile_audit(result: Mapping[str, Any]) -> Dict[str, Any]:
    profile = result.get("profile", {})
    provenance = profile.get("gemm_provenance", {})
    events = profile.get("events", [])
    forbidden_events = [
        event
        for event in events
        if any(
            token in str(event.get("name", "")).lower()
            for token in FORBIDDEN_DEPENDENCIES
        )
    ]
    dispatch = result.get("correctness", {}).get("dispatch", {})
    gaussian = dispatch.get("gaussian") or {}
    pair = dispatch.get("pair") or {}
    return {
        "native_aten_gemm_event_count": int(
            provenance.get("native_aten_gemm_event_count", -1)
        ),
        "all_profiled_gemm_owned_by_flagos_cpp": bool(
            provenance.get("all_profiled_aten_gemm_owned_by_flagos_cpp")
        ),
        "forbidden_profile_events": forbidden_events,
        "gaussian_selected": gaussian.get("selected"),
        "pair_selected": pair.get("selected"),
        "passed": (
            provenance.get("native_aten_gemm_event_count") == 0
            and provenance.get("all_profiled_aten_gemm_owned_by_flagos_cpp") is True
            and not forbidden_events
            and gaussian.get("selected") == "nvidia"
            and pair.get("selected") == "nvidia"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--flaggems-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = {
        "representative": args.representative.resolve(),
        "production": args.production.resolve(),
    }
    loaded = {name: _load_shape(root) for name, root in roots.items()}
    revision_records = {
        json.dumps(suite["revisions"], sort_keys=True)
        for suite, _, _ in loaded.values()
    }
    if len(revision_records) != 1:
        raise AssertionError("representative and production suites used different revisions")

    dependency = _dependency_audit(
        args.library.resolve(), args.flaggems_root.resolve()
    )
    shapes: Dict[str, Any] = {}
    profile_audits: Dict[str, Any] = {}
    expected_library_hashes = set()
    for name, root in roots.items():
        suite, baseline_blocks, candidate_blocks = loaded[name]
        baseline_flat = [value for block in baseline_blocks for value in block]
        candidate_flat = [value for block in candidate_blocks for value in block]
        baseline_summary = suite["aggregate"]["routes"][BASELINE_ROUTE]
        candidate_summary = suite["aggregate"]["routes"][CANDIDATE_ROUTE]
        candidate_result = _candidate_result(root)
        profile_audits[name] = _profile_audit(candidate_result)
        expected_library_hashes.add(
            candidate_result.get("gemm_contract", {}).get("gemm_library_sha256")
        )
        shapes[name] = {
            "workload": suite["workload"],
            "trial_count": suite["trial_count"],
            "sample_count_per_route": len(baseline_flat),
            "torch_fused_step_median_ms": float(statistics.median(baseline_flat)),
            "self_hosted_step_median_ms": float(statistics.median(candidate_flat)),
            "speedup": float(
                statistics.median(baseline_flat) / statistics.median(candidate_flat)
            ),
            "torch_fused_step_p95_ms": float(np.quantile(baseline_flat, 0.95)),
            "self_hosted_step_p95_ms": float(np.quantile(candidate_flat, 0.95)),
            "peak_memory_ratio": float(
                candidate_summary["memory"]["peak_allocated_bytes"]
                / baseline_summary["memory"]["peak_allocated_bytes"]
            ),
            "correctness_passed": bool(suite["correctness"]["passed"]),
            "profile_audit_passed": profile_audits[name]["passed"],
            "suite_sha256": _sha256(root / "suite.json"),
        }

    point_speedup = math.exp(
        statistics.fmean(math.log(row["speedup"]) for row in shapes.values())
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_resamples, dtype=np.float64)
    for index in range(args.bootstrap_resamples):
        per_shape = [
            _resampled_speedup(rng, loaded[name][1], loaded[name][2])
            for name in roots
        ]
        bootstrap[index] = math.exp(
            statistics.fmean(math.log(value) for value in per_shape)
        )
    interval = {
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
    }
    gates = {
        "aggregate_speedup_at_least_1_05": point_speedup >= 1.05,
        "bootstrap_lower_above_1": interval["lower"] > 1.0,
        "no_shape_slower_than_torch": min(row["speedup"] for row in shapes.values()) > 1.0,
        "peak_memory_at_most_1_25x_torch": max(
            row["peak_memory_ratio"] for row in shapes.values()
        )
        <= 1.25,
        "strict_correctness": all(row["correctness_passed"] for row in shapes.values()),
        "profile_dispatch_and_kernel_provenance": all(
            row["profile_audit_passed"] for row in shapes.values()
        ),
        "extension_dependency_audit": dependency["passed"],
        "measured_library_hash_matches": expected_library_hashes
        == {dependency["library_sha256"]},
        "five_trials_fifty_samples": all(
            row["trial_count"] >= 5 and row["sample_count_per_route"] >= 250
            for row in shapes.values()
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "benchmark": "SToFM Stage B V100 self-hosted two-shape acceptance",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validation",
            "verification_status": "VERIFIED" if passed else "FAILED",
            "version_label": "stofm_stageB_v100_acceptance_v1",
        },
        "routes": {"baseline": BASELINE_ROUTE, "candidate": CANDIDATE_ROUTE},
        "revisions": json.loads(next(iter(revision_records))),
        "dependency_audit": dependency,
        "profile_audits": profile_audits,
        "shapes": shapes,
        "aggregate": {
            "method": "equal-weight geometric mean of per-shape median speedups",
            "speedup": point_speedup,
            "latency_reduction_percent": (1.0 - 1.0 / point_speedup) * 100.0,
            "bootstrap_95_ci": interval,
            "bootstrap_method": (
                "fixed workload matrix; paired hierarchical resampling of trial "
                "blocks and CUDA-event samples within each shape"
            ),
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# SToFM Stage B V100 自研 GEMM 验收",
        "",
        f"状态：**{result['status']}**",
        "",
        "| 形状 | Torch fused | FlagOS 自研 | 加速 | P95（Torch → FlagOS） | 显存比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in shapes.items():
        lines.append(
            f"| {name} | {row['torch_fused_step_median_ms']:.4f} ms | "
            f"{row['self_hosted_step_median_ms']:.4f} ms | {row['speedup']:.4f}x | "
            f"{row['torch_fused_step_p95_ms']:.4f} → "
            f"{row['self_hosted_step_p95_ms']:.4f} ms | "
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
    lines.extend(f"- [{'x' if value else ' '}] {key}" for key, value in gates.items())
    args.output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
