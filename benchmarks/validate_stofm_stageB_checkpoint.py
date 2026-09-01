#!/usr/bin/env python3
"""Validate bitwise-equivalent continuous and resumed self-hosted training."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "benchmarks/train_stofm_fake_flagos.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: List[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"checkpoint worker failed ({completed.returncode}):\n{completed.stderr[-4000:]}"
        )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }


def _compare(actual: Any, expected: Any, path: str, rows: List[Dict[str, Any]]) -> None:
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor):
            raise AssertionError(f"checkpoint type mismatch at {path}")
        difference = (
            (actual.float() - expected.float()).abs()
            if actual.numel()
            else torch.zeros((), dtype=torch.float32)
        )
        rows.append(
            {
                "name": path,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "max_abs": float(difference.max()),
                "bitwise_equal": bool(torch.equal(actual, expected)),
            }
        )
        return
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"checkpoint keys differ at {path}")
        for key in actual:
            _compare(actual[key], expected[key], f"{path}.{key}", rows)
        return
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"checkpoint sequence length differs at {path}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare(actual_item, expected_item, f"{path}[{index}]", rows)
        return
    if actual != expected:
        raise AssertionError(f"checkpoint scalar differs at {path}: {actual!r} != {expected!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    library = args.library.resolve()
    environment = os.environ.copy()
    environment["FLAGGEMS_STOFM_SELF_HOSTED_LIBRARY"] = str(library)
    os.environ.update(environment)
    common = [
        sys.executable,
        str(TRAINER),
        "--device",
        "cuda:0",
        "--batch-size",
        "1",
        "--nodes",
        "64",
        "--input-dim",
        "16",
        "--embedding-dim",
        "32",
        "--heads",
        "4",
        "--gaussian-hidden-dim",
        "16",
        "--layers",
        "1",
        "--training-implementation",
        "native",
        "--optimizer",
        "flagos_fused",
        "--gemm-backend",
        "self_hosted",
        "--dispatch-surface",
        "v100_tuned",
        "--no-strict",
        "--no-profile",
    ]
    command_records: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stofm-stageb-checkpoint-") as temporary:
        root = Path(temporary)
        continuous_dir = root / "continuous"
        first_dir = root / "first"
        resumed_dir = root / "resumed"
        command_records.append(
            _run([*common, "--steps", "2", "--output", str(continuous_dir)])
        )
        command_records.append(
            _run([*common, "--steps", "1", "--output", str(first_dir)])
        )
        first_checkpoint = first_dir / "checkpoint-step-001.pt"
        command_records.append(
            _run(
                [
                    *common,
                    "--steps",
                    "1",
                    "--resume",
                    str(first_checkpoint),
                    "--output",
                    str(resumed_dir),
                ]
            )
        )
        continuous_path = continuous_dir / "checkpoint-step-002.pt"
        resumed_path = resumed_dir / "checkpoint-step-002.pt"
        continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
        tensor_rows: List[Dict[str, Any]] = []
        _compare(continuous["model"], resumed["model"], "model", tensor_rows)
        _compare(continuous["optimizer"], resumed["optimizer"], "optimizer", tensor_rows)
        continuous_run = json.loads((continuous_dir / "run.json").read_text(encoding="utf-8"))
        resumed_run = json.loads((resumed_dir / "run.json").read_text(encoding="utf-8"))
        loss_difference = abs(
            float(continuous_run["steps"][1]["loss"])
            - float(resumed_run["steps"][0]["loss"])
        )
        all_bitwise_equal = all(row["bitwise_equal"] for row in tensor_rows)
        max_abs = max((row["max_abs"] for row in tensor_rows), default=0.0)
        passed = (
            continuous["step"] == resumed["step"] == 2
            and continuous["final_parameter_sha256"]
            == resumed["final_parameter_sha256"]
            and loss_difference == 0.0
            and all_bitwise_equal
            and max_abs == 0.0
        )
        result = {
            "schema_version": 1,
            "status": "passed" if passed else "failed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "material_passport": {
                "origin_skill": "experiment-agent",
                "origin_mode": "validation",
                "verification_status": "VERIFIED" if passed else "FAILED",
                "version_label": "stofm_stageB_checkpoint_validation_v1",
            },
            "scope": (
                "Checkpoint semantics only. Strict operator ownership and profiling "
                "are enforced by the two-shape Stage B acceptance suite."
            ),
            "revisions": {
                "stofm": continuous_run["environment"]["stofm_commit"],
                "flaggems": continuous_run["environment"]["flag_gems_commit"],
            },
            "library_sha256": _sha256(library),
            "continuous_steps": 2,
            "resumed_steps": "1 + checkpoint reload + 1",
            "continuous_final_parameter_sha256": continuous["final_parameter_sha256"],
            "resumed_final_parameter_sha256": resumed["final_parameter_sha256"],
            "loss_abs_difference_at_step_2": loss_difference,
            "tensor_count": len(tensor_rows),
            "max_abs": max_abs,
            "all_bitwise_equal": all_bitwise_equal,
            "continuous_checkpoint_sha256": _sha256(continuous_path),
            "resumed_checkpoint_sha256": _sha256(resumed_path),
            "commands": command_records,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
