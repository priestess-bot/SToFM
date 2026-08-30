#!/usr/bin/env python3
"""Run one isolated SToFM PHASE 2 training benchmark route on NVIDIA V100."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FLAGGEMS_ROOT = ROOT.parent / "FlagGems-stofm"
for path in (ROOT, ROOT / "geneformer_001", FLAGGEMS_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.train_stofm_fake_flagos import FakeTrainingConfig, SyntheticSToFMDataset
from model.flagos_optimizer import FlagOSFusedAdamW
from model.flagos_runtime import flagos_training_scope
from model.se2transformer import SToFMForMaskedLM
from model.utils import SToFMConfig


ROUTES: Dict[str, Dict[str, str]] = {
    "torch_scalar": {
        "display_name": "纯 PyTorch 原始算子 + 单张量 AdamW",
        "framework": "torch",
        "training_implementation": "reference",
        "optimizer": "scalar",
    },
    "torch_fused": {
        "display_name": "纯 PyTorch 原始算子 + CUDA fused AdamW",
        "framework": "torch",
        "training_implementation": "reference",
        "optimizer": "torch_fused",
    },
    "flagos_reference_scalar": {
        "display_name": "初始 FlagOS 可微参考算子 + 单张量 AdamW",
        "framework": "flagos",
        "training_implementation": "reference",
        "optimizer": "scalar",
    },
    "flagos_native_scalar": {
        "display_name": "优化后 FlagOS 原生训练算子 + 单张量 AdamW（正式选用）",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "scalar",
    },
    "flagos_reference_fused": {
        "display_name": "FlagOS 可微参考算子 + 逐参数 fused AdamW",
        "framework": "flagos",
        "training_implementation": "reference",
        "optimizer": "flagos_fused",
    },
    "flagos_native_fused": {
        "display_name": "FlagOS 原生训练算子 + 逐参数 fused AdamW（候选）",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "flagos_fused",
    },
}


def _git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _quantile(values: Sequence[float], percentile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), percentile))


def _summarize(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample sequence")
    return {
        "samples_ms": [float(value) for value in values],
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "std_ms": float(statistics.pstdev(values)),
        "min_ms": float(min(values)),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "max_ms": float(max(values)),
    }


def _tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _model_digest(model: torch.nn.Module) -> str:
    return _tensor_digest(dict(model.named_parameters()))


def _model_config(args: argparse.Namespace, route: Mapping[str, str]) -> SToFMConfig:
    flagos = route["framework"] == "flagos"
    return SToFMConfig(
        num_hidden_layers=args.layers,
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.embedding_dim,
        num_attention_heads=args.heads,
        gaussian_hidden_dim=args.gaussian_hidden_dim,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        flagos_mode="optimized" if flagos else "torch",
        flagos_backend="nvidia" if flagos else "torch",
        flagos_attention_backend="nvidia" if flagos else "torch",
        flagos_training_implementation=route["training_implementation"],
    )


def _build_model_and_optimizer(
    args: argparse.Namespace, route: Mapping[str, str], device: torch.device
):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = SToFMForMaskedLM(_model_config(args, route)).to(device).train()
    if route["optimizer"] == "flagos_fused":
        optimizer = FlagOSFusedAdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=False,
            fused=route["optimizer"] == "torch_fused",
        )
    return model, optimizer


def _loss(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]):
    outputs = model(**batch)
    return outputs, outputs["loss"] + outputs["pair_loss"]


def _dispatch_snapshot(model: SToFMForMaskedLM) -> Dict[str, Any]:
    gaussian = model.model.gaussian.last_flagos_dispatch
    pair = model.model.encoder.layers[0].self_attn.last_flagos_dispatch
    return {
        "gaussian": asdict(gaussian) if gaussian is not None else None,
        "pair": asdict(pair) if pair is not None else None,
    }


def _optimizer_state_by_name(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Dict[str, torch.Tensor]]:
    result = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        tensors = {
            key: value.detach().cpu().clone()
            for key, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        if tensors:
            result[name] = tensors
    return result


def _correctness_step(
    model: SToFMForMaskedLM,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    output: Path,
) -> Dict[str, Any]:
    initial_sha = _model_digest(model)
    optimizer.zero_grad(set_to_none=True)
    outputs, total_loss = _loss(model, batch)
    total_loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    dispatch = _dispatch_snapshot(model)
    scalar_losses = {
        "total": float(total_loss.detach().cpu()),
        "mcm": float(outputs["loss"].detach().cpu()),
        "pdr": float(outputs["pair_loss"].detach().cpu()),
    }
    optimizer.step()
    torch.cuda.synchronize()
    updated_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    payload = {
        "losses": scalar_losses,
        "gradients": gradients,
        "updated_parameters": updated_parameters,
        "optimizer_state": _optimizer_state_by_name(model, optimizer),
    }
    snapshot_path = output / "first_step.pt"
    torch.save(payload, snapshot_path)
    return {
        "initial_parameter_sha256": initial_sha,
        "updated_parameter_sha256": _tensor_digest(updated_parameters),
        "gradient_sha256": _tensor_digest(gradients),
        "losses": scalar_losses,
        "dispatch": dispatch,
        "snapshot": snapshot_path.name,
        "gradient_tensor_count": len(gradients),
    }


def _untimed_step(model, optimizer, batch) -> None:
    optimizer.zero_grad(set_to_none=True)
    _, total_loss = _loss(model, batch)
    total_loss.backward()
    optimizer.step()


def _timed_step(model, optimizer, batch) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    full_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    full_start.record()
    _, total_loss = _loss(model, batch)
    forward_end.record()
    total_loss.backward()
    backward_end.record()
    optimizer.step()
    optimizer_end.record()
    optimizer_end.synchronize()
    result = {
        "forward_ms": full_start.elapsed_time(forward_end),
        "backward_ms": forward_end.elapsed_time(backward_end),
        "optimizer_ms": backward_end.elapsed_time(optimizer_end),
        "step_ms": full_start.elapsed_time(optimizer_end),
    }
    del total_loss
    return result


def _benchmark(
    model,
    optimizer,
    batch,
    *,
    repetitions: int,
) -> Dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    samples = []
    for _ in range(repetitions):
        samples.append(_timed_step(model, optimizer, batch))
    metrics = {
        key: _summarize([sample[key] for sample in samples])
        for key in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms")
    }
    return {
        "metrics": metrics,
        "raw_samples": samples,
        "memory": {
            "allocated_before_samples_bytes": int(allocated_before),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    }


def _kernel_summary(trace_path: Path) -> Dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    counts: Dict[str, int] = {}
    durations: Dict[str, float] = {}
    for event in trace.get("traceEvents", []):
        if "kernel" not in str(event.get("cat", "")).lower():
            continue
        name = str(event.get("name", "unknown"))
        counts[name] = counts.get(name, 0) + 1
        durations[name] = durations.get(name, 0.0) + float(event.get("dur", 0.0))
    ranked = sorted(durations, key=durations.get, reverse=True)
    return {
        "kernel_event_count": sum(counts.values()),
        "unique_kernel_count": len(counts),
        "top_kernels": [
            {
                "name": name,
                "count": counts[name],
                "total_duration_us": durations[name],
            }
            for name in ranked[:30]
        ],
    }


def _profile_step(model, optimizer, batch, output: Path) -> Dict[str, Any]:
    trace_path = output / "training_trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        _untimed_step(model, optimizer, batch)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    events = []
    for event in profiler.key_averages():
        if event.self_device_time_total <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_device_us": float(event.self_device_time_total),
                "self_cpu_us": float(event.self_cpu_time_total),
            }
        )
    events.sort(key=lambda item: item["self_device_us"], reverse=True)
    profile = {
        "trace": trace_path.name,
        "events": events,
        "kernel_summary": _kernel_summary(trace_path),
    }
    (output / "profile_summary.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    return profile


def _environment() -> Dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "device": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=384)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 1 or args.repetitions < 1 or args.trial < 1:
        raise ValueError("warmup, repetitions, and trial must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PHASE 2 V100 worker requires CUDA")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    route = ROUTES[args.route]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            nodes=args.nodes,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            gaussian_hidden_dim=args.gaussian_hidden_dim,
            layers=args.layers,
            learning_rate=args.learning_rate,
            dropout=0.0,
        ),
        device,
    )
    batch = dataset.batch()
    batch_sha = _tensor_digest(batch)
    correctness_model, correctness_optimizer = _build_model_and_optimizer(args, route, device)
    warmup_model, warmup_optimizer = _build_model_and_optimizer(args, route, device)
    benchmark_model, benchmark_optimizer = _build_model_and_optimizer(args, route, device)
    initial_hashes = {
        _model_digest(correctness_model),
        _model_digest(warmup_model),
        _model_digest(benchmark_model),
    }
    if len(initial_hashes) != 1:
        raise AssertionError("deterministic model initialization failed")

    scope = (
        flagos_training_scope(mode="optimized", strict=True)
        if route["framework"] == "flagos"
        else nullcontext(None)
    )
    with scope as runtime:
        correctness = _correctness_step(
            correctness_model, correctness_optimizer, batch, output
        )
        del correctness_model, correctness_optimizer
        gc.collect()
        effective_warmup = max(args.warmup, args.repetitions)
        for _ in range(effective_warmup):
            _untimed_step(warmup_model, warmup_optimizer, batch)
        torch.cuda.synchronize()
        del warmup_model, warmup_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        timing = _benchmark(
            benchmark_model,
            benchmark_optimizer,
            batch,
            repetitions=args.repetitions,
        )
        profile = (
            _profile_step(benchmark_model, benchmark_optimizer, batch, output)
            if args.profile
            else {"status": "not_requested"}
        )
        runtime_dispatch = asdict(runtime) if runtime is not None else None

    median_step_ms = timing["metrics"]["step_ms"]["median_ms"]
    timing["throughput"] = {
        "nodes_per_second": args.batch_size * args.nodes * 1000.0 / median_step_ms,
        "pairs_per_second": args.batch_size * args.nodes**2 * 1000.0 / median_step_ms,
    }
    result = {
        "schema_version": 1,
        "benchmark": "SToFM PHASE 2 V100 FP32 training",
        "route_id": args.route,
        "route": route,
        "trial": args.trial,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "stofm_phase2_v100_training_worker_v1",
            "upstream_dependencies": ["deps/flagos-training.lock.json"],
        },
        "revisions": {
            "stofm": _git_sha(ROOT),
            "flaggems": _git_sha(FLAGGEMS_ROOT),
        },
        "environment": _environment(),
        "workload": {
            "precision": "fp32",
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "input_dim": args.input_dim,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "layers": args.layers,
            "objective": "MCM + PDR",
            "data": "deterministic synthetic tensors",
        },
        "protocol": {
            "requested_warmup": args.warmup,
            "effective_warmup": max(args.warmup, args.repetitions),
            "repetitions": args.repetitions,
            "timer": "CUDA events; one synchronization after each complete sample",
            "timed_region": "forward + backward + optimizer; zero_grad excluded",
            "compile_policy": (
                "a disposable model warms every measured optimizer step index; "
                "timing restarts from identical initial weights and optimizer state"
            ),
        },
        "batch_sha256": batch_sha,
        "runtime_dispatch": runtime_dispatch,
        "correctness": correctness,
        "timing": timing,
        "profile": profile,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "route": args.route, "output": str(output)}))


if __name__ == "__main__":
    main()
