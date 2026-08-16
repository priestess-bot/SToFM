#!/usr/bin/env python3
"""Probe the frozen unoptimized FlagOS source on a MUSA target without fallback."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", default=None)
    args = parser.parse_args()

    stock_root = args.stock_root.resolve()
    source_root = stock_root / "src"
    if not (source_root / "flag_gems").is_dir():
        raise FileNotFoundError(f"frozen FlagOS source package does not exist: {source_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "stock_probe.log"
    result = {
        "schema_version": 1,
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("musa-stock-probe-%Y%m%dT%H%M%SZ"),
        "stage": "frozen_unoptimized_flagos",
        "status": "unavailable",
        "stock_revision": args.expected_revision or _git_revision(stock_root),
        "runtime": {"torch": torch.__version__},
    }
    try:
        import torch_musa

        result["runtime"].update(
            {
                "torch_musa": torch_musa.__version__,
                "musa_available": bool(torch.musa.is_available()),
                "device": torch.musa.get_device_name(0) if torch.musa.is_available() else None,
            }
        )
        sys.path.insert(0, str(source_root))
        import flag_gems

        # A successful import is not enough: this is the exact narrow scope
        # used by the SToFM comparison, and it must register without fallback.
        with flag_gems.use_gems(include=["addmm", "baddbmm", "bmm", "softmax"]):
            registered = tuple(sorted(str(item) for item in flag_gems.all_registered_ops()))
        result.update(
            {
                "status": "available",
                "registered_aten_ops": list(registered),
                "reason": "frozen source imported and created the requested ATen scope",
            }
        )
        log_path.write_text("Frozen FlagOS MUSA probe completed.\n", encoding="utf-8")
    except Exception as exc:
        trace = traceback.format_exc()
        log_path.write_text(trace, encoding="utf-8")
        result.update(
            {
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "log": log_path.name,
            }
        )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "available":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
