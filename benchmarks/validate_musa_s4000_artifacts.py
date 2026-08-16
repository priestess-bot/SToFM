#!/usr/bin/env python3
"""Independently validate a completed MTT S4000 formal evidence directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List


EXPECTED_STOFM = "e2c6de9ec902bee5d67a4861b4ef6716a58e0cc4"
EXPECTED_FLAGGEMS = "832c46df4073215d416406181484f9b44594aff2"
EXPECTED_STOCK_FLAGGEMS = "03bf364ede763d573d5c30124d554283a209ab85"
EXPECTED_INITIAL_LIBRARY = "006d5e256060342f1fb188f91e623fed0baaa0746928d710a43de63efb1cf590"
EXPECTED_OPTIMIZED_LIBRARY = "a7beac88e8d4b7b999b3620b13234ded848aee22d1a479f66ce9f9744a8e2313"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_close(actual: float, expected: float, *, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"aggregate mismatch: actual={actual}, expected={expected}")


def _check_samples(measurement: Dict[str, Any], context: str) -> None:
    samples = measurement["samples_ms"]
    host_samples = measurement["host_samples_ms"]
    if measurement["sample_count"] != len(samples):
        raise AssertionError(f"{context}: sample_count does not match device samples")
    if len(samples) != len(host_samples):
        raise AssertionError(f"{context}: device and host sample counts differ")
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise AssertionError(f"{context}: device samples must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in host_samples):
        raise AssertionError(f"{context}: host samples must be finite and positive")
    _assert_close(measurement["p50_ms"], statistics.median(samples))


def _check_sources(
    result: Dict[str, Any], *, expected_library: str, context: str
) -> None:
    sources = result["sources"]
    if sources["stofm_revision"] != EXPECTED_STOFM:
        raise AssertionError(f"{context}: unexpected SToFM revision")
    if sources["flaggems_revision"] != EXPECTED_FLAGGEMS:
        raise AssertionError(f"{context}: unexpected FlagGems revision")
    if sources["musa_library_sha256"] != expected_library:
        raise AssertionError(f"{context}: unexpected MUSA library hash")
    if not result["measurement"].get("inference_mode"):
        raise AssertionError(f"{context}: timing was not recorded in inference mode")
    if result["measurement"].get("compile_included"):
        raise AssertionError(f"{context}: compile time must be excluded")


def _check_matrix(
    result: Dict[str, Any],
    *,
    expected_library: str,
    expected_rows: int,
    context: str,
) -> None:
    _check_sources(result, expected_library=expected_library, context=context)
    if len(result["results"]) != expected_rows:
        raise AssertionError(f"{context}: unexpected operator row count")
    for row in result["results"]:
        if set(row) < {
            "operator",
            "precision",
            "nodes",
            "validation",
            "torch_reference",
            "flagos_candidate",
        }:
            raise AssertionError(f"{context}: incomplete operator result")
        max_error = row["validation"]["errors"]["max_abs_error"]
        if not math.isfinite(max_error) or max_error < 0:
            raise AssertionError(f"{context}: invalid correctness error")
        _check_samples(row["torch_reference"], f"{context}/torch/{row['operator']}")
        _check_samples(row["flagos_candidate"], f"{context}/flagos/{row['operator']}")


def _check_model(result: Dict[str, Any], context: str) -> None:
    _check_sources(
        result, expected_library=EXPECTED_OPTIMIZED_LIBRARY, context=context
    )
    if result["workload"]["nodes"] != 1050 or result["workload"]["layers"] != 4:
        raise AssertionError(f"{context}: unexpected primary workload")
    measured = [row for row in result["results"] if row["status"] == "measured"]
    unavailable = [row for row in result["results"] if row["status"] == "unavailable"]
    if len(measured) != 4 or len(unavailable) != 1:
        raise AssertionError(f"{context}: expected four measured and one unavailable stage")
    for row in measured:
        if row["validation"]["status"] != "passed":
            raise AssertionError(f"{context}/{row['stage']}: correctness gate failed")
        _check_samples(row, f"{context}/{row['stage']}")
    stock_aten = unavailable[0]
    if stock_aten["samples_ms"]:
        raise AssertionError(f"{context}: unavailable ATen stage contains substituted timing")


def _median_trial_p50(results: Iterable[Dict[str, Any]], selector) -> float:
    return statistics.median(selector(result) for result in results)


def validate(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "manifest.json")
    expected_top_hashes = {
        "summary_sha256": _sha256(root / "summary.json"),
        "operator_summary_sha256": _sha256(root / "operator_summary.csv"),
        "report_sha256": _sha256(root / "report.md"),
    }
    for key, digest in expected_top_hashes.items():
        if manifest[key] != digest:
            raise AssertionError(f"manifest hash mismatch for {key}")

    trial_dirs = sorted(path for path in root.glob("trial-*") if path.is_dir())
    if len(trial_dirs) != 5 or summary["trial_count"] != 5:
        raise AssertionError("formal evidence must contain five trials")
    listed_results = sorted(manifest["trial_result_files"])
    discovered_results = sorted(
        str(path.relative_to(root)) for path in root.glob("trial-*/**/result.json")
    )
    if listed_results != discovered_results or len(discovered_results) != 15:
        raise AssertionError("formal manifest must enumerate exactly 15 trial results")

    initial_trials: List[Dict[str, Any]] = []
    optimized_trials: List[Dict[str, Any]] = []
    model_trials: List[Dict[str, Any]] = []
    for index, trial_dir in enumerate(trial_dirs, start=1):
        initial = _read_json(trial_dir / "initial-flagos-operators" / "result.json")
        optimized = _read_json(trial_dir / "optimized-operator-matrix" / "result.json")
        model = _read_json(trial_dir / "end-to-end-model" / "result.json")
        if {initial["trial"], optimized["trial"], model["trial"]} != {index}:
            raise AssertionError(f"trial-{index:02d}: inconsistent trial identifiers")
        _check_matrix(
            initial,
            expected_library=EXPECTED_INITIAL_LIBRARY,
            expected_rows=2,
            context=f"trial-{index:02d}/initial",
        )
        _check_matrix(
            optimized,
            expected_library=EXPECTED_OPTIMIZED_LIBRARY,
            expected_rows=24,
            context=f"trial-{index:02d}/optimized",
        )
        _check_model(model, f"trial-{index:02d}/model")
        initial_trials.append(initial)
        optimized_trials.append(optimized)
        model_trials.append(model)

    for comparison in summary["primary_operator_comparison"]:
        operator = comparison["operator"]

        def matrix_p50(result: Dict[str, Any], implementation: str) -> float:
            row = next(
                row
                for row in result["results"]
                if row["operator"] == operator
                and row["precision"] == "fp32"
                and row["nodes"] == 1050
            )
            return row[implementation]["p50_ms"]

        _assert_close(
            comparison["torch_p50_ms"],
            _median_trial_p50(
                optimized_trials, lambda result: matrix_p50(result, "torch_reference")
            ),
        )
        _assert_close(
            comparison["initial_flagos_p50_ms"],
            _median_trial_p50(
                initial_trials, lambda result: matrix_p50(result, "flagos_candidate")
            ),
        )
        _assert_close(
            comparison["optimized_flagos_p50_ms"],
            _median_trial_p50(
                optimized_trials, lambda result: matrix_p50(result, "flagos_candidate")
            ),
        )

    stock = summary["frozen_upstream_flagos"]
    if stock["stock_revision"] != EXPECTED_STOCK_FLAGGEMS:
        raise AssertionError("unexpected frozen upstream FlagOS revision")
    if stock["status"] != "unavailable" or "0 active drivers" not in stock["reason"]:
        raise AssertionError("frozen upstream FlagOS unavailability was not preserved")

    sensitive_fragments = (
        "ssh -p ",
        "password:",
        "BEGIN OPENSSH PRIVATE KEY",
    )
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".log", ".md"}:
            content = path.read_text(encoding="utf-8", errors="replace")
            if any(fragment in content for fragment in sensitive_fragments):
                raise AssertionError(f"sensitive connection data found in {path}")

    verification_path = root / "verification.json"
    tracked_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != verification_path
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "checks": {
            "independent_trials": 5,
            "trial_result_files": 15,
            "operator_measurement_rows": 130,
            "model_measured_rows": 20,
            "source_revisions_verified": True,
            "library_hashes_verified": True,
            "inference_mode_verified": True,
            "correctness_gates_verified": True,
            "sample_vectors_verified": True,
            "aggregates_recomputed": True,
            "unavailable_baseline_not_substituted": True,
            "sensitive_connection_data_absent": True,
        },
        "source_revisions": summary["source_revisions"],
        "file_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in tracked_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.artifact_dir)
    output = args.output or args.artifact_dir / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
