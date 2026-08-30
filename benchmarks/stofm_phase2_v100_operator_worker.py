#!/usr/bin/env python3
"""Benchmark one SToFM PHASE 2 training operator implementation on V100."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FLAGGEMS_ROOT = ROOT.parent / "FlagGems-stofm"
for path in (ROOT, FLAGGEMS_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from flag_gems.experimental_ops import stofm_gaussian_pair_bias, stofm_pair_attention
from flag_gems.experimental_ops._stofm_common import (
    gaussian_pair_bias_dense,
    pair_attention_reference,
)
from model.flagos_runtime import flagos_training_scope


IMPLEMENTATIONS = {
    "torch": "纯 PyTorch 原始实现",
    "flagos_reference": "初始 FlagOS 可微参考实现",
    "flagos_native": "FlagOS 原生训练实现",
}


def _git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _quantile(values: Sequence[float], percentile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "samples_ms": [float(value) for value in values],
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "std_ms": float(statistics.pstdev(values)),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _clone_leaves(values: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [value.detach().clone().requires_grad_(True) for value in values]


def _errors(actual: Sequence[torch.Tensor], reference: Sequence[torch.Tensor]) -> Dict[str, Any]:
    rows = []
    for actual_tensor, reference_tensor in zip(actual, reference):
        difference = (actual_tensor.float() - reference_tensor.float()).abs()
        relative = difference / reference_tensor.float().abs().clamp_min(1e-6)
        rows.append(
            {
                "max_abs": float(difference.max()),
                "max_rel": float(relative.max()),
                "mean_abs": float(difference.mean()),
            }
        )
    return {
        "tensor_count": len(rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "max_rel": max(row["max_rel"] for row in rows),
        "per_tensor": rows,
    }


class GaussianCase:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        b, n, k, h = args.batch_size, args.nodes, args.gaussian_hidden_dim, args.heads
        distances = torch.rand((b, n, n), generator=generator, device=device)
        distances[:, torch.arange(n), torch.arange(n)] = 0.0
        scale_k = k**-0.5
        values = [
            distances,
            torch.tensor([[0.8]], device=device),
            torch.tensor([0.1], device=device),
            torch.linspace(-1.5, 1.5, k, device=device).reshape(1, k),
            torch.linspace(0.45, 1.55, k, device=device).reshape(1, k),
            torch.randn((k, k), generator=generator, device=device) * scale_k,
            torch.randn((k,), generator=generator, device=device) * scale_k,
            torch.randn((h, k), generator=generator, device=device) * scale_k,
            torch.randn((h,), generator=generator, device=device) * scale_k,
        ]
        self.inputs = _clone_leaves(values)
        self.zero_mask = distances.eq(0.0)
        self.grad_output = torch.randn(
            (b, h, n, n), generator=generator, device=device
        )

    def call(self, implementation: str, inputs: Sequence[torch.Tensor]):
        if implementation == "torch":
            return gaussian_pair_bias_dense(*inputs, self.zero_mask)
        return stofm_gaussian_pair_bias(
            *inputs,
            zero_mask=self.zero_mask,
            backend="nvidia",
            training=True,
            training_implementation=(
                "native" if implementation == "flagos_native" else "reference"
            ),
        )

    def run(self, implementation: str, timed: bool):
        inputs = _clone_leaves(self.inputs)
        if timed:
            start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            start.record()
        output = self.call(implementation, inputs)
        if timed:
            forward_end.record()
        gradients = torch.autograd.grad(output, inputs, self.grad_output)
        if not timed:
            return (output.detach(),), tuple(gradient.detach() for gradient in gradients)
        backward_end.record()
        backward_end.synchronize()
        return {
            "forward_ms": start.elapsed_time(forward_end),
            "backward_ms": forward_end.elapsed_time(backward_end),
            "total_ms": start.elapsed_time(backward_end),
        }


class PairCase:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        b, h, n = args.batch_size, args.heads, args.nodes
        d = args.embedding_dim // h
        query = torch.randn((b, h, n, d), generator=generator, device=device) * d**-0.5
        values = [
            query,
            torch.randn((b, h, n, d), generator=generator, device=device),
            torch.randn((b, h, n, d), generator=generator, device=device),
            torch.randn((b, h, n, n), generator=generator, device=device) * 0.1,
        ]
        self.inputs = _clone_leaves(values)
        self.padding = torch.zeros((b, n), dtype=torch.bool, device=device)
        self.padding[:, -1] = True
        self.grad_context = torch.randn(
            (b, h, n, d), generator=generator, device=device
        )
        self.grad_pair = torch.randn(
            (b, h, n, n), generator=generator, device=device
        )

    def call(self, implementation: str, inputs: Sequence[torch.Tensor]):
        if implementation == "torch":
            return pair_attention_reference(
                *inputs,
                key_padding_mask=self.padding,
                dropout_p=0.0,
                training=True,
                scale=1.0,
                return_pair=True,
                return_weights=False,
            )
        return stofm_pair_attention(
            *inputs,
            key_padding_mask=self.padding,
            dropout_p=0.0,
            training=True,
            scale=1.0,
            return_pair=True,
            return_weights=False,
            backend="nvidia",
            training_implementation=(
                "native" if implementation == "flagos_native" else "reference"
            ),
        )

    def run(self, implementation: str, timed: bool):
        inputs = _clone_leaves(self.inputs)
        if timed:
            start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            start.record()
        context, pair, _ = self.call(implementation, inputs)
        if timed:
            forward_end.record()
        gradients = torch.autograd.grad(
            (context, pair), inputs, (self.grad_context, self.grad_pair)
        )
        if not timed:
            return (
                context.detach(),
                pair.detach(),
            ), tuple(gradient.detach() for gradient in gradients)
        backward_end.record()
        backward_end.synchronize()
        return {
            "forward_ms": start.elapsed_time(forward_end),
            "backward_ms": forward_end.elapsed_time(backward_end),
            "total_ms": start.elapsed_time(backward_end),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("gaussian", "pair"), required=True)
    parser.add_argument("--implementation", choices=tuple(IMPLEMENTATIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("operator benchmark requires CUDA")
    if args.warmup < 1 or args.repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = GaussianCase(args, device) if args.operator == "gaussian" else PairCase(args, device)

    reference_outputs, reference_gradients = case.run("torch", timed=False)
    torch.cuda.synchronize()
    scope = (
        flagos_training_scope(mode="optimized", strict=True)
        if args.implementation != "torch"
        else nullcontext(None)
    )
    with scope:
        actual_outputs, actual_gradients = case.run(args.implementation, timed=False)
        correctness = {
            "outputs": _errors(actual_outputs, reference_outputs),
            "gradients": _errors(actual_gradients, reference_gradients),
        }
        correctness["passed"] = (
            correctness["outputs"]["max_abs"] <= 5e-4
            and correctness["gradients"]["max_abs"] <= 2e-3
        )
        if not correctness["passed"]:
            raise AssertionError("operator correctness threshold failed")
        del actual_outputs, actual_gradients, reference_outputs, reference_gradients
        gc.collect()
        for _ in range(args.warmup):
            case.run(args.implementation, timed=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = [case.run(args.implementation, timed=True) for _ in range(args.repetitions)]

    metrics = {
        key: _summary([sample[key] for sample in samples])
        for key in ("forward_ms", "backward_ms", "total_ms")
    }
    result = {
        "schema_version": 1,
        "benchmark": "SToFM PHASE 2 V100 operator training",
        "operator": args.operator,
        "implementation": args.implementation,
        "display_name": IMPLEMENTATIONS[args.implementation],
        "trial": args.trial,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "stofm_phase2_v100_operator_worker_v1",
        },
        "revisions": {
            "stofm": _git_sha(ROOT),
            "flaggems": _git_sha(FLAGGEMS_ROOT),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "workload": {
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "head_dim": args.embedding_dim // args.heads,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "precision": "fp32",
        },
        "protocol": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "timer": "CUDA events",
            "backward_objective": "fixed explicit output gradients",
        },
        "correctness": correctness,
        "metrics": metrics,
        "raw_samples": samples,
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "operator": args.operator, "implementation": args.implementation}))


if __name__ == "__main__":
    main()
