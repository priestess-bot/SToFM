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
from model.flagos_runtime import (
    flagos_training_scope,
    prepare_flagos_vendor_training_fusion,
)
from model.se2transformer import SToFMForMaskedLM
from model.utils import SToFMConfig


ROUTES: Dict[str, Dict[str, Any]] = {
    "torch_scalar": {
        "display_name": "纯 PyTorch 原始算子 + 单张量 AdamW",
        "framework": "torch",
        "training_implementation": "reference",
        "optimizer": "scalar",
        "gemm_backend": "triton",
    },
    "torch_fused": {
        "display_name": "纯 PyTorch 原始算子 + CUDA fused AdamW",
        "framework": "torch",
        "training_implementation": "reference",
        "optimizer": "torch_fused",
        "gemm_backend": "triton",
    },
    "torch_compile_fused": {
        "display_name": "PyTorch compile 辅助对照 + CUDA fused AdamW",
        "framework": "torch_compile",
        "training_implementation": "reference",
        "optimizer": "torch_fused",
        "gemm_backend": "triton",
    },
    "flagos_reference_scalar": {
        "display_name": "初始 FlagOS 可微参考算子 + 单张量 AdamW",
        "framework": "flagos",
        "training_implementation": "reference",
        "optimizer": "scalar",
        "gemm_backend": "triton",
    },
    "flagos_native_scalar": {
        "display_name": "优化后 FlagOS 原生训练算子 + 单张量 AdamW（正式选用）",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "scalar",
        "gemm_backend": "triton",
    },
    "flagos_reference_fused": {
        "display_name": "FlagOS 可微参考算子 + 逐参数 fused AdamW",
        "framework": "flagos",
        "training_implementation": "reference",
        "optimizer": "flagos_fused",
        "gemm_backend": "triton",
    },
    "flagos_native_fused": {
        "display_name": "FlagOS 原生训练算子 + 逐参数 fused AdamW（候选）",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "flagos_fused",
        "gemm_backend": "triton",
    },
    "flagos_vendor_reference_scalar": {
        "display_name": "FlagOS 可微参考算子 + Vendor GEMM + 单张量 AdamW",
        "framework": "flagos",
        "training_implementation": "reference",
        "optimizer": "scalar",
        "gemm_backend": "vendor",
    },
    "flagos_vendor_native_scalar": {
        "display_name": "FlagOS 原生训练算子 + Vendor GEMM + 单张量 AdamW",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "scalar",
        "gemm_backend": "vendor",
    },
    "flagos_vendor_native_fused": {
        "display_name": "FlagOS 原生训练算子 + Vendor GEMM + fused AdamW",
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "flagos_fused",
        "gemm_backend": "vendor",
    },
    "flagos_vendor_native_fused_v100_tuned": {
        "display_name": (
            "FlagOS V100 调优：Vendor GEMM/BMM + Gaussian 融合反向 + 多张量 AdamW"
        ),
        "framework": "flagos",
        "training_implementation": "native",
        "optimizer": "flagos_fused",
        "gemm_backend": "vendor",
        "gaussian_training_implementation": "native",
        "attention_training_implementation": "reference",
        # An empty ATen list deliberately leaves non-critical pointwise ops on
        # PyTorch's device kernels while Vendor owns every GEMM/BMM.  The full
        # registrar remains a separate route for regression and gap accounting.
        "aten_include": (),
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
    return _tensor_digest(dict(_unwrap_model(model).named_parameters()))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


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
        flagos_gaussian_training_implementation=route.get(
            "gaussian_training_implementation", route["training_implementation"]
        ),
        flagos_attention_training_implementation=route.get(
            "attention_training_implementation", route["training_implementation"]
        ),
        flagos_gemm_backend=route.get("gemm_backend", "triton"),
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
    if route["framework"] == "torch_compile":
        model = torch.compile(model, dynamic=False, fullgraph=False, mode="default")
    return model, optimizer


def _loss(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]):
    outputs = model(**batch)
    return outputs, outputs["loss"] + outputs["pair_loss"]


def _dispatch_snapshot(model: SToFMForMaskedLM) -> Dict[str, Any]:
    model = _unwrap_model(model)
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
    raw_model = _unwrap_model(model)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in raw_model.named_parameters()
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
        for name, parameter in raw_model.named_parameters()
    }
    payload = {
        "losses": scalar_losses,
        "gradients": gradients,
        "updated_parameters": updated_parameters,
        "optimizer_state": _optimizer_state_by_name(raw_model, optimizer),
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
        # Application clocks cannot be locked on the shared host. A fixed
        # device-side spin immediately before every sample keeps P-state from
        # becoming a route/order confound; it is outside the CUDA-event region.
        torch.cuda._sleep(50_000_000)
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


def _gemm_provenance(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Audit the dispatcher-level GEMM owner from profiler operator events."""
    native_names = {
        "aten::mm",
        "aten::addmm",
        "aten::bmm",
        "aten::baddbmm",
    }
    aten_events = [event for event in events if event.get("name") in native_names]
    dispatcher_ownership = {}
    for operator in sorted(native_names):
        table = torch._C._dispatch_dump_table(operator)
        cuda_entry = next(
            (line for line in table.splitlines() if line.startswith("CUDA:")), "missing"
        )
        autograd_entry = next(
            (
                line
                for line in table.splitlines()
                if line.startswith("AutogradCUDA:")
            ),
            "missing",
        )
        dispatcher_ownership[operator] = {
            "cuda_entry": cuda_entry,
            "autograd_cuda_entry": autograd_entry,
            "vendor_cpp_cuda_kernel": "stofm_vendor_gemm.cu" in cuda_entry,
        }
    native_events = [
        event
        for event in aten_events
        if not dispatcher_ownership[event["name"]]["vendor_cpp_cuda_kernel"]
    ]
    vendor_events = [
        event
        for event in events
        if str(event.get("name", "")).startswith("flagos_stofm_vendor::")
    ]
    vendor_kernels = [
        event
        for event in events
        if any(
            marker in str(event.get("name", "")).lower()
            for marker in ("sgemm", "gemmex", "gemmstrided", "cublas")
        )
    ]
    return {
        "aten_gemm_operator_events": aten_events,
        "aten_gemm_operator_event_count": len(aten_events),
        "dispatcher_ownership": dispatcher_ownership,
        "native_aten_gemm_events": native_events,
        "native_aten_gemm_event_count": len(native_events),
        "vendor_dispatch_events": vendor_events,
        "vendor_dispatch_event_count": len(vendor_events),
        "vendor_kernel_events": vendor_kernels,
        "native_gemm_absent": not native_events,
        "all_profiled_aten_gemm_owned_by_vendor_cpp": all(
            dispatcher_ownership[event["name"]]["vendor_cpp_cuda_kernel"]
            for event in aten_events
        ),
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
        "gemm_provenance": _gemm_provenance(events),
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


def _prime_cuda_clock(device: torch.device, milliseconds: int = 500) -> Dict[str, Any]:
    """Run a short compute-bound kernel before every route's measured phase.

    This host does not permit application-clock locking.  Without priming, a
    launch-bound route can enter timing at 135 MHz while a GEMM-heavy route
    enters at boost clocks, which is a framework-order bias rather than model
    performance.  The prime is outside the CUDA-event timed region and applies
    identically to every route.
    """
    size = 1024
    left = torch.randn(size, size, device=device)
    right = torch.randn(size, size, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = 0
    elapsed = 0.0
    while elapsed < milliseconds:
        start.record()
        for _ in range(10):
            left = torch.mm(left, right)
            left.mul_(1.0 / size)
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end)
        iterations += 10
    return {
        "target_ms": milliseconds,
        "actual_ms": float(elapsed),
        "gemm_iterations": iterations,
        "timed_region": False,
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
    parser.add_argument(
        "--cuda-profiler-range",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bracket one additional steady step with cudaProfilerStart/Stop for Nsight",
    )
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

    preparation_records = []
    for preparation_name, preparation_model in (
        ("correctness", correctness_model),
        ("warmup", warmup_model),
        ("benchmark", benchmark_model),
    ):
        elapsed = prepare_flagos_vendor_training_fusion(preparation_model, batch)
        if elapsed is not None:
            preparation_records.append(
                {"model": preparation_name, "elapsed_ms": float(elapsed)}
            )
    preparation_ms = (
        float(sum(row["elapsed_ms"] for row in preparation_records))
        if preparation_records
        else None
    )

    scope = (
        flagos_training_scope(
            mode="optimized",
            strict=True,
            gemm_backend=route.get("gemm_backend", "triton"),
            include=route.get("aten_include"),
        )
        if route["framework"] == "flagos"
        else nullcontext(None)
    )
    with scope as runtime:
        clock_prime = _prime_cuda_clock(device)
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
        cuda_profiler_range = {"status": "not_requested"}
        if args.cuda_profiler_range:
            cudart = torch.cuda.cudart()
            start_status = int(cudart.cudaProfilerStart())
            _untimed_step(benchmark_model, benchmark_optimizer, batch)
            torch.cuda.synchronize()
            stop_status = int(cudart.cudaProfilerStop())
            cuda_profiler_range = {
                "status": "captured"
                if start_status == 0 and stop_status == 0
                else "failed",
                "start_status": start_status,
                "stop_status": stop_status,
                "steps": 1,
            }
        if (
            route.get("gemm_backend") == "vendor"
            and profile.get("gemm_provenance", {}).get("native_aten_gemm_event_count", 0)
            > 0
        ):
            raise RuntimeError(
                "Vendor route profile observed native ATen GEMM events; see profile_summary.json"
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
            "per_sample_clock_control": {
                "method": "torch.cuda._sleep",
                "cycles": 50_000_000,
                "inside_timed_region": False,
                "reason": "application clock locking is unavailable on this shared V100 host",
            },
            "compile_policy": (
                "a disposable model warms every measured optimizer step index; "
                "timing restarts from identical initial weights and optimizer state"
            ),
            "vendor_training_fusion_preparation_ms": preparation_ms,
            "vendor_training_fusion_preparation_records": preparation_records,
            "vendor_training_fusion_prepared_outside_full_aten_scope": bool(preparation_records),
            "cuda_clock_prime": clock_prime,
        },
        "batch_sha256": batch_sha,
        "runtime_dispatch": runtime_dispatch,
        "gemm_contract": {
            "backend": route.get("gemm_backend", "triton"),
            "native_torch_gemm_allowed": False
            if route.get("gemm_backend") == "vendor"
            else None,
            "vendor_library": (
                runtime_dispatch.get("vendor_gemm_library")
                if runtime_dispatch is not None
                else None
            ),
        },
        "preparation": {
            "gaussian_backward_fusion_ms": preparation_ms,
            "gaussian_backward_fusion_ready": bool(preparation_records),
            "records": preparation_records,
        },
        "correctness": correctness,
        "timing": timing,
        "profile": profile,
        "cuda_profiler_range": cuda_profiler_range,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "route": args.route, "output": str(output)}))


if __name__ == "__main__":
    main()
