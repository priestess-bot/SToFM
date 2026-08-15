"""Verify the committed V100 evidence for registered SToFM custom operators.

This checker is deliberately read-only.  It verifies evidence integrity and
execution provenance; it does not reproduce latency numbers or make a target
hardware claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from write_r2_checksums import verify_manifest


STOCK_STAGES = {
    "pure_pytorch_reference",
    "unoptimized_flagos_lifecycle",
    "unoptimized_flagos_steady",
}
OPTIMIZED_STAGES = {
    "gaussian_registered_operator_only",
    "pair_score_registered_operator_only",
    "registered_operators_only_combined",
    "registered_operators_with_flagos_aten_steady",
    "registered_operators_with_flagos_aten_lifecycle",
}
EXPECTED_TRACES = {
    "gaussian_registered_operator_only": ("flagos_stofm::gaussian_pair_bias",),
    "pair_score_registered_operator_only": ("flagos_stofm::pair_score_epilogue",),
    "registered_operators_only_combined": (
        "flagos_stofm::gaussian_pair_bias",
        "flagos_stofm::pair_score_epilogue",
    ),
    "registered_operators_with_flagos_aten_steady": (
        "flagos_stofm::gaussian_pair_bias",
        "flagos_stofm::pair_score_epilogue",
    ),
    "registered_operators_with_flagos_aten_lifecycle": (
        "flagos_stofm::gaussian_pair_bias",
        "flagos_stofm::pair_score_epilogue",
    ),
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_source_root(result: Dict[str, Any]) -> None:
    source = Path(result["flaggems_source"]["source_path"]).resolve()
    imported = Path(result["flaggems_source"]["imported_package"]).resolve()
    _expect(source.name == "src", f"unexpected FlagGems source directory: {source}")
    _expect(imported.is_relative_to(source), "FlagGems import escaped the requested source root")


def _verify_result(result: Dict[str, Any], *, role: str) -> None:
    expected_stages = STOCK_STAGES if role == "stock" else OPTIMIZED_STAGES
    _expect(result["role"] == role, f"expected {role} worker result")
    _expect(result["schema_version"] == 3, "unexpected worker schema version")
    _expect(result["benchmark_suite"] == "registered_ops", "not a registered-operator suite")
    _expect(result["measurement"] == {
        "timer": "cuda_events",
        "warmup": 10,
        "repetitions": 30,
        "calls_per_sample": 5,
        "compile_included": False,
        "tf32": False,
        "inference_mode": True,
    }, "measurement controls differ from the strict V100 protocol")
    _expect(result["runtime"]["device"] == "Tesla V100-SXM2-16GB", "unexpected benchmark GPU")
    _expect(result["runtime"]["capability"] == [7, 0], "unexpected V100 capability")
    _verify_source_root(result)

    rows = {row["stage"]: row for row in result["results"]}
    _expect(set(rows) == expected_stages, f"unexpected stage set for {role}")
    for stage, row in rows.items():
        _expect(row["status"] == "measured", f"{stage} was not measured")
        _expect(row["validation"]["status"] == "passed", f"{stage} failed numerical validation")
        _expect(row["sample_count"] == 30, f"{stage} has an unexpected sample count")
        _expect(len(row["samples_ms"]) == 30, f"{stage} has incomplete raw samples")
        _expect(row["calls_per_sample"] == 5, f"{stage} has an unexpected call count")
        expected_trace = EXPECTED_TRACES.get(stage, ())
        _expect(
            tuple(row["custom_operator_trace"]) == expected_trace,
            f"{stage} custom-operator profiler trace differs from the required route",
        )
        if stage in {
            "gaussian_registered_operator_only",
            "pair_score_registered_operator_only",
            "registered_operators_only_combined",
        }:
            _expect(
                not row["dispatch"]["runtime"]["active"],
                f"{stage} must isolate custom operators from ATen dispatch",
            )


def verify_suite(root: Path) -> Dict[str, Any]:
    """Verify a completed FP32 or FP16 registered-operator benchmark tree."""
    root = root.resolve()
    suite = _read_json(root / "suite.json")
    _expect(suite["schema_version"] == 3, "unexpected aggregate schema version")
    _expect(suite["benchmark_suite"] == "registered_ops", "unexpected aggregate suite")
    _expect(suite["run_count"] == 3, "the protocol requires three independent trials")
    _expect(len(suite["trials"]) == 3, "trial metadata is incomplete")

    commits: list[Tuple[str, str, str]] = []
    references = set()
    for trial in suite["trials"]:
        stock = _read_json(root / trial["stock_result"])
        optimized = _read_json(root / trial["optimized_result"])
        _verify_result(stock, role="stock")
        _verify_result(optimized, role="optimized")
        _expect(
            stock["reference"]["last_hidden_state_sha256"]
            == optimized["reference"]["last_hidden_state_sha256"],
            "pure PyTorch reference hashes differ across package environments",
        )
        references.add(stock["reference"]["last_hidden_state_sha256"])
        commits = [
            *commits,
            (
                stock["commits"]["stofm"],
                stock["commits"]["flaggems"],
                optimized["commits"]["flaggems"],
            ),
        ]

    _expect(len(set(commits)) == 1, "commit provenance differs across trials")
    _expect(verify_manifest(root) >= 20, "checksum manifest has incomplete evidence")
    return {
        "status": "passed",
        "precision": suite["precision"],
        "trial_count": suite["run_count"],
        "reference_hash_count": len(references),
        "commits": next(iter(set(commits))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([verify_suite(directory) for directory in args.directories], indent=2))


if __name__ == "__main__":
    main()
