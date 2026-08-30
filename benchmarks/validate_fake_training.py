#!/usr/bin/env python3
"""Validate checkpoint resume against an uninterrupted FlagOS training run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_tensor_difference(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Tuple[float, str | None]:
    maximum = 0.0
    worst: str | None = None
    if left.keys() != right.keys():
        missing = sorted(set(left) ^ set(right))
        raise AssertionError(f"state keys differ: {missing[:5]}")
    for name in left:
        first, second = left[name], right[name]
        if torch.is_tensor(first) and torch.is_tensor(second):
            difference = (first.detach().cpu() - second.detach().cpu()).abs()
            value = float(difference.max()) if difference.numel() else 0.0
            if value > maximum:
                maximum, worst = value, name
        elif first != second:
            raise AssertionError(f"non-tensor state differs at {name}")
    return maximum, worst


def _checkpoint_difference(left: Path, right: Path) -> Dict[str, Any]:
    first = torch.load(left, map_location="cpu", weights_only=False)
    second = torch.load(right, map_location="cpu", weights_only=False)
    model_max, model_worst = _max_tensor_difference(first["model"], second["model"])
    optimizer_max = 0.0
    optimizer_worst = None
    first_state = first["optimizer"]["state"]
    second_state = second["optimizer"]["state"]
    if first_state.keys() != second_state.keys():
        raise AssertionError("optimizer state keys differ")
    for state_id in first_state:
        current, worst = _max_tensor_difference(first_state[state_id], second_state[state_id])
        if current > optimizer_max:
            optimizer_max, optimizer_worst = current, f"{state_id}:{worst}"
    return {
        "model_max_abs": model_max,
        "model_worst_tensor": model_worst,
        "optimizer_max_abs": optimizer_max,
        "optimizer_worst_tensor": optimizer_worst,
        "left_step": int(first.get("step", -1)),
        "right_step": int(second.get("step", -1)),
    }


def validate(resume_dir: Path, continuous_dir: Path, atol: float) -> Dict[str, Any]:
    resumed = _load_json(resume_dir / "run.json")
    continuous = _load_json(continuous_dir / "run.json")
    if resumed.get("status") != "passed" or continuous.get("status") != "passed":
        raise AssertionError("both runs must have status=passed")
    if resumed.get("strict") is not True or continuous.get("strict") is not True:
        raise AssertionError("both runs must use strict FlagOS mode")
    for name, result in (("resumed", resumed), ("continuous", continuous)):
        fallback = result["operator_inventory"]["fallback_compute_ops"]
        if fallback:
            raise AssertionError(f"{name} run has compute fallbacks: {fallback}")

    resumed_step = resumed["steps"][-1]
    target_step = int(resumed_step["step"])
    continuous_steps = {
        int(row["step"]): row for row in continuous["steps"]
    }
    if target_step not in continuous_steps:
        raise AssertionError(f"continuous run has no step {target_step}")
    continuous_step = continuous_steps[target_step]
    metric_differences = {
        key: abs(float(resumed_step[key]) - float(continuous_step[key]))
        for key in ("loss", "mcm_loss", "pdr_loss", "max_grad")
    }
    if any(value > atol for value in metric_differences.values()):
        raise AssertionError(f"metric differences exceed atol={atol}: {metric_differences}")

    checkpoint_difference = _checkpoint_difference(
        Path(resumed["checkpoint"]), Path(continuous["checkpoint"])
    )
    if checkpoint_difference["left_step"] != checkpoint_difference["right_step"]:
        raise AssertionError("checkpoint step numbers differ")
    if checkpoint_difference["model_max_abs"] > atol:
        raise AssertionError(
            "model resume difference exceeds "
            f"atol={atol}: {checkpoint_difference['model_max_abs']}"
        )
    if checkpoint_difference["optimizer_max_abs"] > atol:
        raise AssertionError(
            "optimizer resume difference exceeds "
            f"atol={atol}: {checkpoint_difference['optimizer_max_abs']}"
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "VERIFIED",
            "version_label": "stofm_flagos_training_validation_v1",
            "upstream_dependencies": [
                str(resume_dir / "run.json"),
                str(continuous_dir / "run.json"),
            ],
        },
        "method": "deterministic resume versus uninterrupted run",
        "absolute_tolerance": atol,
        "resumed_run": str(resume_dir),
        "continuous_run": str(continuous_dir),
        "compared_step": target_step,
        "metric_absolute_differences": metric_differences,
        "checkpoint_difference": checkpoint_difference,
        "conclusion": (
            "checkpoint restore is reproducible within the declared FP32 GPU tolerance; "
            "byte-level parameter hashes are not required because CUDA reduction order "
            "can differ by a few ulps"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate(args.resumed.resolve(), args.continuous.resolve(), args.atol)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    report = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {result['material_passport']['origin_date']}",
        "- Verification Status: VERIFIED",
        "- Version Label: stofm_flagos_training_validation_v1",
        "",
        "# SToFM FlagOS checkpoint 恢复验证",
        "",
        f"- 结论：**{result['status']}**",
        f"- 对比 step：{result['compared_step']}",
        f"- FP32 绝对误差门槛：`{result['absolute_tolerance']}`",
        f"- loss 最大差异：`{max(result['metric_absolute_differences'].values()):.3g}`",
        f"- 模型参数最大绝对差异：`{result['checkpoint_difference']['model_max_abs']:.3g}`",
        f"- 优化器状态最大绝对差异：`{result['checkpoint_difference']['optimizer_max_abs']:.3g}`",
        "",
        result["conclusion"],
    ]
    (args.output / "validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
