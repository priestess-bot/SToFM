#!/usr/bin/env python3
"""Run a small, deterministic SToFM MCM+PDR training loop with FlagGems.

The workload intentionally bypasses Scanpy/Geneformer and generates the model
boundary tensors in memory.  It is a training integration smoke test, not a
biological benchmark.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "geneformer_001") not in sys.path:
    sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.flagos_runtime import (
    STOFM_TRAINING_ALLOWLIST,
    STOFM_TRAINING_METADATA_OPS,
    flagos_training_scope,
    prepare_flagos_training_fusion,
)
from model.flagos_optimizer import FlagOSFusedAdamW
from model.se2transformer import SToFMForMaskedLM
from model.utils import SToFMConfig


@dataclass(frozen=True)
class FakeTrainingConfig:
    seed: int = 20260830
    steps: int = 10
    batch_size: int = 2
    nodes: int = 12
    input_dim: int = 8
    embedding_dim: int = 16
    heads: int = 4
    gaussian_hidden_dim: int = 8
    layers: int = 2
    learning_rate: float = 1e-3
    dropout: float = 0.0


class SyntheticSToFMDataset:
    """A fixed tensor dataset matching SToFM's training interface."""

    def __init__(self, config: FakeTrainingConfig, device: torch.device):
        cpu_generator = torch.Generator(device="cpu").manual_seed(config.seed)
        b, n, d = config.batch_size, config.nodes, config.input_dim
        h = config.embedding_dim
        self.token_embeddings = torch.randn(b, n, d, generator=cpu_generator)
        self.attn_bias = torch.rand(b, n, n, generator=cpu_generator)
        self.attn_bias[:, torch.arange(n), torch.arange(n)] = 0.0
        self.token_types = torch.zeros(b, n, dtype=torch.long)
        self.token_types[:, 0] = 1  # CLS
        self.token_types[:, -1] = 3  # padding

        # A deterministic teacher-shaped target keeps the smoke meaningful while
        # remaining independent of any downloaded biological data.
        labels = torch.randn(b, n, h, generator=cpu_generator)
        labels[:, 0, :] = -100.0
        labels[:, -1, :] = -100.0
        self.labels = labels
        pair_labels = torch.randn(b, n, n, generator=cpu_generator)
        pair_labels[:, 0, :] = -100.0
        pair_labels[:, -1, :] = -100.0
        pair_labels[:, :, 0] = -100.0
        pair_labels[:, :, -1] = -100.0
        self.pair_labels = pair_labels

        self.device = device
        self._move_to_device()

    def _move_to_device(self) -> None:
        for name in (
            "token_embeddings",
            "attn_bias",
            "token_types",
            "labels",
            "pair_labels",
        ):
            setattr(self, name, getattr(self, name).to(self.device))

    def batch(self) -> Dict[str, torch.Tensor]:
        return {
            "token_embeddings": self.token_embeddings,
            "attn_bias": self.attn_bias,
            "token_types": self.token_types,
            "labels": self.labels,
            "pair_labels": self.pair_labels,
        }


class FlagOSAdamW(torch.optim.AdamW):
    """AdamW configuration that avoids unimplemented foreach FlagGems ops."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float):
        super().__init__(params, lr=lr, foreach=False, fused=False, capturable=False)


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _flag_gems_architecture() -> Dict[str, Any]:
    """Capture whether FlagGems has an architecture-specific tuning profile."""
    result: Dict[str, Any] = {
        "has_specialization": None,
        "arch": None,
        "current_arch_path": None,
        "note": "unavailable",
    }
    try:
        from flag_gems.runtime import backend

        event = backend.BackendArchEvent()
        result.update(
            {
                "has_specialization": bool(getattr(event, "has_arch", False)),
                "arch": getattr(event, "arch", None),
                "current_arch_path": getattr(event, "current_arch_path", None),
                "note": (
                    "architecture-specific profile is active"
                    if getattr(event, "has_arch", False)
                    else "generic FlagGems path; no architecture-specific profile"
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - vendor/runtime dependent
        result["note"] = f"architecture probe failed: {type(exc).__name__}"
    return result


def _environment(device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "flag_gems_source": None,
        "flag_gems_commit": _git_sha(ROOT.parent / "FlagGems-stofm"),
        "stofm_commit": _git_sha(ROOT),
        "flag_gems_architecture": _flag_gems_architecture()
        if device.type == "cuda"
        else None,
    }
    if device.type == "cuda":
        try:
            import flag_gems

            result["flag_gems_source"] = str(Path(flag_gems.__file__).resolve())
        except ImportError:
            pass
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["compute_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def _kernel_evidence(trace_path: Path) -> Dict[str, Any]:
    """Summarize raw CUDA kernel labels from a Chrome trace.

    The labels are evidence for follow-up inspection, not a replacement for
    Nsight attribution.  FlagGems Triton launches are commonly named
    ``*_kernel`` while native ATen launches carry an ``at::native`` prefix.
    """
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    counts = Counter(
        str(event.get("name"))
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X" and event.get("cat") == "kernel" and event.get("name")
    )
    unique_names = sorted(counts)
    candidate_flaggems = sorted(
        name
        for name in unique_names
        if "_kernel" in name
        and not name.startswith("void at::")
        and not name.startswith("at::")
    )
    return {
        "status": "captured",
        "event_count": int(sum(counts.values())),
        "unique_kernel_names": unique_names,
        "kernel_counts": dict(sorted(counts.items())),
        "candidate_flaggems_kernel_names": candidate_flaggems,
        "attribution_note": (
            "Raw profiler labels only; candidate names should be confirmed with "
            "Nsight Systems/Compute before making a kernel-level performance claim."
        ),
    }


def _trace_operator_attribution(trace_path: Path) -> Dict[str, Any]:
    """Join profiler CPU operator events to CUDA kernels by External id."""
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}

    kernels_by_external_id: Dict[Any, List[str]] = defaultdict(list)
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is not None:
            kernels_by_external_id[external_id].append(str(event.get("name", "")))

    summary: Dict[str, Dict[str, Any]] = {}
    for event in trace.get("traceEvents", []):
        name = str(event.get("name", ""))
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        if not name.startswith("aten::"):
            continue
        operator = name.removeprefix("aten::")
        external_id = event.get("args", {}).get("External id")
        kernel_names = kernels_by_external_id.get(external_id, [])
        row = summary.setdefault(
            operator,
            {
                "cpu_event_count": 0,
                "mapped_cpu_event_count": 0,
                "flaggems_kernel_event_count": 0,
                "native_kernel_event_count": 0,
                "kernel_names": Counter(),
            },
        )
        row["cpu_event_count"] += 1
        if kernel_names:
            row["mapped_cpu_event_count"] += 1
        for kernel_name in kernel_names:
            row["kernel_names"][kernel_name] += 1
            if kernel_name.startswith("void at::") or kernel_name.startswith("at::"):
                row["native_kernel_event_count"] += 1
            else:
                row["flaggems_kernel_event_count"] += 1

    normalized: Dict[str, Any] = {}
    for operator, row in sorted(summary.items()):
        normalized[operator] = {
            **{key: value for key, value in row.items() if key != "kernel_names"},
            "kernel_names": dict(sorted(row["kernel_names"].items())),
            "classification": (
                "partial_native_fallback"
                if row["native_kernel_event_count"] and row["flaggems_kernel_event_count"]
                else "native_fallback"
                if row["native_kernel_event_count"]
                else "flaggems_kernel"
                if row["flaggems_kernel_event_count"]
                else "unmapped"
            ),
        }
    return {
        "status": "captured",
        "operators": normalized,
        "join_key": "traceEvents[*].args['External id']",
        "native_kernel_rule": "kernel name starts with 'void at::' or 'at::'",
        "attribution_note": (
            "This is a deterministic profiler join. Native/FlagGems labels are retained "
            "for audit; Nsight remains the final kernel attribution tool."
        ),
    }


def _flaggems_log_summary(log_path: Path) -> Dict[str, Any]:
    """Extract the operator function names written by FlagGems' debug logger."""
    if not log_path.is_file():
        return {"status": "not_requested"}
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    names = sorted(
        set(
            match.group(1)
            for match in re.finditer(r"flag_gems\.ops\.([^.\s]+)", text)
        )
    )
    return {
        "status": "captured",
        "line_count": len(text.splitlines()),
        "operator_functions": names,
    }


def _config(args: argparse.Namespace) -> SToFMConfig:
    return SToFMConfig(
        num_hidden_layers=args.layers,
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.embedding_dim,
        num_attention_heads=args.heads,
        gaussian_hidden_dim=args.gaussian_hidden_dim,
        dropout=args.dropout,
        attention_dropout=args.dropout,
        activation_dropout=args.dropout,
        flagos_mode="optimized",
        flagos_backend="nvidia" if args.device.startswith("cuda") else "torch",
        flagos_attention_backend="nvidia" if args.device.startswith("cuda") else "torch",
        flagos_training_implementation=args.training_implementation,
        flagos_gaussian_training_implementation=args.training_implementation,
        flagos_attention_training_implementation=(
            "reference"
            if args.dispatch_surface == "v100_tuned"
            and args.gemm_backend == "vendor"
            else args.training_implementation
        ),
        flagos_gemm_backend=args.gemm_backend,
    )


def _check_finite(model: torch.nn.Module) -> Tuple[bool, float]:
    max_grad = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return False, float("inf")
        max_grad = max(max_grad, float(parameter.grad.detach().abs().max().cpu()))
    return True, max_grad


def _profile_step(model, optimizer, batch, output_dir: Path) -> Dict[str, Any]:
    if not hasattr(torch, "profiler"):
        return {"status": "unavailable", "reason": "torch.profiler is unavailable"}
    activities = [torch.profiler.ProfilerActivity.CPU]
    if batch["token_embeddings"].device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # Profile a complete step, then restore the exact training state so the
    # checkpoint remains at the requested step count rather than step+1.
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if batch["token_embeddings"].device.type == "cuda" else None
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs["loss"] + outputs["pair_loss"]
            loss.backward()
            optimizer.step()
            if batch["token_embeddings"].device.type == "cuda":
                torch.cuda.synchronize()
    finally:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    trace_path = output_dir / "training_trace.json"
    profile.export_chrome_trace(str(trace_path))
    kernel_evidence = _kernel_evidence(trace_path)
    operator_attribution = _trace_operator_attribution(trace_path)
    events = []
    for event in profile.key_averages():
        if (
            not event.key.startswith("aten::")
            and not event.key.startswith("Optimizer")
            and not event.key.startswith("flagos_stofm::")
        ):
            continue
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_us": round(float(event.self_cpu_time_total), 3),
                "self_device_us": round(float(event.self_device_time_total), 3),
            }
        )
    events.sort(key=lambda item: item["self_device_us"], reverse=True)
    (output_dir / "training_profile.json").write_text(
        json.dumps(
            {
                "status": "captured",
                "events": events,
                "kernel_evidence": kernel_evidence,
                "operator_attribution": operator_attribution,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "captured",
        "trace": str(trace_path),
        "profile": str(output_dir / "training_profile.json"),
        "event_count": len(events),
        "kernel_evidence": kernel_evidence,
        "operator_attribution": operator_attribution,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--training-implementation",
        choices=("reference", "native"),
        default="reference",
    )
    parser.add_argument(
        "--optimizer",
        choices=("scalar", "flagos_fused"),
        default="scalar",
    )
    parser.add_argument(
        "--gemm-backend",
        choices=("triton", "vendor", "self_hosted"),
        default="triton",
        help=(
            "FlagOS-owned GEMM dispatcher; self_hosted uses the BLAS-free "
            "V100 CUDA implementation"
        ),
    )
    parser.add_argument(
        "--dispatch-surface",
        choices=("full", "v100_tuned"),
        default="full",
        help="full FlagOS registrar or the V100 measured tuned surface",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.nodes < 3:
        raise ValueError("steps and batch-size must be positive; nodes must be at least 3")
    if args.strict and not args.profile:
        raise ValueError("strict FlagOS training requires --profile for fallback auditing")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but torch.cuda is unavailable")
    if args.training_implementation == "native" and not args.device.startswith("cuda"):
        raise ValueError("native training implementation currently requires CUDA")
    if args.optimizer == "flagos_fused" and not args.device.startswith("cuda"):
        raise ValueError("FlagOS fused AdamW currently requires CUDA")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.strict and device.type != "cuda":
        raise ValueError(
            "strict FlagOS training is only defined for CUDA in this phase; "
            "use --no-strict for a CPU semantic reference run"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    config = _config(args)
    model = SToFMForMaskedLM(config).to(device).train()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            nodes=args.nodes,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            gaussian_hidden_dim=args.gaussian_hidden_dim,
            layers=args.layers,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
        ),
        device,
    )
    batch = dataset.batch()
    optimizer = (
        FlagOSFusedAdamW(model.parameters(), lr=args.learning_rate)
        if args.optimizer == "flagos_fused"
        else FlagOSAdamW(model.parameters(), lr=args.learning_rate)
    )
    start_step = 0
    resumed_from = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        resumed_from = str(resume_path)
    initial_parameter_sha256 = _parameter_sha256(model)
    losses: List[Dict[str, Any]] = []
    dispatch_record: Optional[Dict[str, Any]] = None
    profile_result: Dict[str, Any] = {"status": "not_requested"}
    start = time.perf_counter()

    flaggems_log_path = output_dir / "flaggems_ops.log"
    training_mode = "optimized" if device.type == "cuda" else "torch"
    preparation_ms = prepare_flagos_training_fusion(model, batch)
    training_include = () if args.dispatch_surface == "v100_tuned" else None
    with flagos_training_scope(
        mode=training_mode,
        strict=args.strict,
        record=args.profile,
        record_path=str(flaggems_log_path) if args.profile else None,
        gemm_backend=args.gemm_backend,
        include=training_include,
    ) as runtime:
        dispatch_record = asdict(runtime)
        for step in range(start_step, start_step + args.steps):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            total_loss = outputs["loss"] + outputs["pair_loss"]
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            total_loss.backward()
            gradients_finite, max_grad = _check_finite(model)
            if not gradients_finite:
                raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            losses.append(
                {
                    "step": step,
                    "loss": float(total_loss.detach().cpu()),
                    "mcm_loss": float(outputs["loss"].detach().cpu()),
                    "pdr_loss": float(outputs["pair_loss"].detach().cpu()),
                    "max_grad": max_grad,
                    "step_ms": (time.perf_counter() - step_start) * 1000.0,
                    "gaussian_dispatch": asdict(model.model.gaussian.last_flagos_dispatch)
                    if model.model.gaussian.last_flagos_dispatch is not None
                    else None,
                    "pair_dispatch": asdict(model.model.encoder.layers[0].self_attn.last_flagos_dispatch)
                    if model.model.encoder.layers[0].self_attn.last_flagos_dispatch is not None
                    else None,
                }
            )
        if args.profile:
            profile_result = _profile_step(model, optimizer, batch, output_dir)

    final_parameter_sha256 = _parameter_sha256(model)
    end_step = start_step + args.steps
    checkpoint_path = output_dir / f"checkpoint-step-{end_step:03d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": end_step,
            "seed": args.seed,
            "config": config.to_dict(),
            "initial_parameter_sha256": initial_parameter_sha256,
            "final_parameter_sha256": final_parameter_sha256,
        },
        checkpoint_path,
    )
    (output_dir / "loss.csv").write_text(
        "step,loss,mcm_loss,pdr_loss,max_grad,step_ms\n"
        + "\n".join(
            f"{row['step']},{row['loss']:.10g},{row['mcm_loss']:.10g},"
            f"{row['pdr_loss']:.10g},{row['max_grad']:.10g},{row['step_ms']:.6f}"
            for row in losses
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(device), indent=2), encoding="utf-8"
    )
    observed = set()
    operator_attribution: Dict[str, Any] = {"status": "not_requested", "operators": {}}
    if profile_result.get("status") == "captured":
        profile_data = json.loads((output_dir / "training_profile.json").read_text())
        observed = {
            event["name"].removeprefix("aten::")
            for event in profile_data["events"]
            if event["name"].startswith("aten::") and event["self_device_us"] > 0.1
        }
        operator_attribution = profile_data.get(
            "operator_attribution", operator_attribution
        )
    allowed = set(STOFM_TRAINING_ALLOWLIST)
    metadata = set(STOFM_TRAINING_METADATA_OPS)
    # The profile uses schema names; map the aliases used by the manifest.
    aliases = {
        "native_layer_norm": "layer_norm",
        "native_layer_norm_backward": "layer_norm_backward",
        "linalg_vector_norm": "vector_norm",
        "_softmax": "softmax",
        "_softmax_backward_data": "softmax_backward",
        "embedding_backward": "embedding_dense_backward",
        "div": "true_divide",
        "div_": "true_divide_",
        "where": "where_self",
        "masked_fill_": "masked_fill_",
        "linear": "addmm",
        "matmul": "mm",
        "rsub": "rsub_tensor",
        "square": "square",
        "lerp_": "lerp_tensor_",
    }
    normalized_observed = {aliases.get(name, name) for name in observed}
    fallback_reasons: Dict[str, List[str]] = defaultdict(list)
    approved_native_compute_ops = set()
    gemm_names = {"mm", "addmm", "bmm", "baddbmm"}
    gemm_dispatcher_ownership = {}
    if args.gemm_backend in {"vendor", "self_hosted"}:
        for name in sorted(gemm_names):
            table = torch._C._dispatch_dump_table(f"aten::{name}")
            cuda_entry = next(
                (line for line in table.splitlines() if line.startswith("CUDA:")),
                "missing",
            )
            source_marker = (
                "stofm_vendor_gemm.cu"
                if args.gemm_backend == "vendor"
                else "stofm_self_hosted_gemm.cu"
            )
            gemm_dispatcher_ownership[name] = {
                "cuda_entry": cuda_entry,
                "flagos_cpp_cuda_kernel": source_marker in cuda_entry,
            }
    flaggems_execution_ops = set()
    native_execution_ops = set()
    partial_execution_ops = set()
    unmapped_execution_ops = set()
    for raw_name in observed:
        normalized_name = aliases.get(raw_name, raw_name)
        if normalized_name in metadata:
            continue
        tuned_native_approved = (
            args.dispatch_surface == "v100_tuned"
            and (
                normalized_name not in gemm_names
                or gemm_dispatcher_ownership.get(normalized_name, {}).get(
                    "flagos_cpp_cuda_kernel", False
                )
            )
        )
        if normalized_name not in allowed and not tuned_native_approved:
            fallback_reasons[normalized_name].append("not in training allowlist")
        details = operator_attribution.get("operators", {}).get(raw_name)
        if not details:
            if tuned_native_approved:
                approved_native_compute_ops.add(normalized_name)
            else:
                unmapped_execution_ops.add(normalized_name)
            continue
        if details.get("flaggems_kernel_event_count", 0):
            flaggems_execution_ops.add(normalized_name)
        if details.get("native_kernel_event_count", 0):
            native_execution_ops.add(normalized_name)
        if details.get("classification") == "partial_native_fallback":
            partial_execution_ops.add(normalized_name)
        if details.get("classification") == "unmapped":
            unmapped_execution_ops.add(normalized_name)
        if details.get("native_kernel_event_count", 0):
            classification = details.get("classification", "native_fallback")
            if tuned_native_approved:
                approved_native_compute_ops.add(normalized_name)
            else:
                fallback_reasons[normalized_name].append(
                    f"{classification}: {details['native_kernel_event_count']} native CUDA kernel event(s)"
                )
        elif details.get("mapped_cpu_event_count", 0) == 0:
            if tuned_native_approved:
                approved_native_compute_ops.add(normalized_name)
            else:
                fallback_reasons[normalized_name].append(
                    "operator-to-kernel join was unmapped"
                )
    fallback_compute_ops = sorted(fallback_reasons)
    operator_inventory = {
        "observed_profile_ops": sorted(observed),
        "normalized_compute_ops": sorted(normalized_observed),
        "requested_flaggems_ops": (
            list(dispatch_record.get("registered_aten_ops", ()))
            if dispatch_record is not None
            else []
        ),
        "metadata_ops": list(STOFM_TRAINING_METADATA_OPS),
        "fallback_compute_ops": fallback_compute_ops,
        "fallback_reasons": {
            name: sorted(set(reasons))
            for name, reasons in sorted(fallback_reasons.items())
        },
        "approved_native_compute_ops": sorted(approved_native_compute_ops),
        "gemm_dispatcher_ownership": gemm_dispatcher_ownership,
        "operator_attribution": operator_attribution,
        "execution_summary": {
            "observed_compute_ops": sorted(normalized_observed - metadata),
            "flaggems_kernel_ops": sorted(flaggems_execution_ops),
            "native_kernel_ops": sorted(native_execution_ops),
            "partial_native_fallback_ops": sorted(partial_execution_ops),
            "unmapped_ops": sorted(unmapped_execution_ops),
            "flaggems_kernel_coverage": (
                len(flaggems_execution_ops)
                / max(1, len(normalized_observed - metadata))
            ),
        },
        "strict_pass": not fallback_compute_ops,
        "notes": {
            "cosine_embedding_loss": "replaced by an equivalent masked cosine decomposition",
            "adamw": (
                "FlagOS packed multi-tensor AdamW; up to 64 tensors per CUDA launch"
                if args.optimizer == "flagos_fused"
                and args.gemm_backend in {"vendor", "self_hosted"}
                else "FlagGems per-parameter fused AdamW; not a cross-parameter foreach kernel"
                if args.optimizer == "flagos_fused"
                else "single-tensor AdamW updates"
            ),
            "amp": "not part of this FP32 smoke run",
            "v100_tuned_surface": (
                "non-GEMM CUDA kernels are explicitly approved; GEMM CUDA ownership "
                "must resolve to the selected FlagOS C++/CUDA dispatcher"
                if args.dispatch_surface == "v100_tuned"
                else "not active"
            ),
        },
    }
    (output_dir / "operator_inventory.json").write_text(
        json.dumps(operator_inventory, indent=2), encoding="utf-8"
    )
    (output_dir / "fallback_report.json").write_text(
        json.dumps(
            {
                "status": (
                    "clean_with_approved_native"
                    if not fallback_compute_ops and approved_native_compute_ops
                    else "clean"
                    if not fallback_compute_ops
                    else "fallbacks_detected"
                ),
                "compute_ops": fallback_compute_ops,
                "reasons": {
                    name: sorted(set(reasons))
                    for name, reasons in sorted(fallback_reasons.items())
                },
                "native_kernel_ops": sorted(native_execution_ops),
                "approved_native_compute_ops": sorted(approved_native_compute_ops),
                "gemm_dispatcher_ownership": gemm_dispatcher_ownership,
                "partial_native_fallback_ops": sorted(partial_execution_ops),
                "unmapped_ops": sorted(unmapped_execution_ops),
                "metadata_ops": sorted(metadata & normalized_observed),
                "host_sync_ops": sorted(
                    {"item", "_local_scalar_dense"} & normalized_observed
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.strict and fallback_compute_ops:
        raise RuntimeError(
            "strict FlagOS training observed unapproved compute ops: "
            + ", ".join(fallback_compute_ops)
        )

    result = {
        "schema_version": 1,
        "status": "passed",
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("fake-training-%Y%m%dT%H%M%SZ"),
        "mode": "flagos_training",
        "flagos_route": (
            (
                f"FlagGems ATen training + {args.gemm_backend} GEMM"
                if training_mode == "optimized"
                else "Torch reference"
            )
        ),
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "origin_date": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_status": "UNVERIFIED",
            "version_label": "stofm_flagos_training_v1",
            "upstream_dependencies": [
                "deps/flagos-training.lock.json",
                "requirements/flagos-training-v100.txt",
            ],
        },
        "strict": args.strict,
        "duration_ms": (time.perf_counter() - start) * 1000.0,
        "start_step": start_step,
        "end_step": end_step,
        "resumed_from": resumed_from,
        "environment": _environment(device),
        "config": vars(args),
        "runtime_dispatch": dispatch_record,
        "preparation": {
            "gaussian_backward_fusion_ms": preparation_ms,
            "gaussian_backward_fusion_ready": preparation_ms is not None,
        },
        "dispatch_surface": args.dispatch_surface,
        "steps": losses,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": final_parameter_sha256,
        "checkpoint": str(checkpoint_path),
        "profile": profile_result,
        "flaggems_log": (
            str(flaggems_log_path) if flaggems_log_path.is_file() else None
        ),
        "flaggems_log_summary": _flaggems_log_summary(flaggems_log_path),
        "operator_inventory": operator_inventory,
    }
    result["timing_summary"] = {
        "first_step_ms": losses[0]["step_ms"] if losses else None,
        "steady_step_p50_ms": float(
            np.median([row["step_ms"] for row in losses[1:]])
        )
        if len(losses) > 1
        else None,
        "note": "first step includes any lazy FlagGems compilation; steady p50 excludes the first recorded step",
    }
    result["config"]["output"] = str(output_dir)
    (output_dir / "run.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def _markdown_report(result: Dict[str, Any]) -> str:
    steps = result["steps"]
    first = steps[0]["loss"]
    last = steps[-1]["loss"]
    passport = result.get("material_passport", {})
    environment = result.get("environment", {})
    architecture = environment.get("flag_gems_architecture") or {}
    execution_summary = result.get("operator_inventory", {}).get(
        "execution_summary", {}
    )
    output_root = Path(result.get("config", {}).get("output", ".")).resolve()

    def display_path(value: Any) -> str:
        if not value:
            return "not recorded"
        try:
            return str(Path(str(value)).resolve().relative_to(output_root))
        except (ValueError, OSError):
            return str(value)

    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {passport.get('origin_date', 'not recorded')}",
        f"- Verification Status: {passport.get('verification_status', 'UNVERIFIED')}",
        "- Version Label: stofm_flagos_training_v1",
        "",
        "# SToFM FlagOS 假数据训练报告",
        "",
        "## 结论",
        "",
        f"- 状态：**{result['status']}**",
        f"- 模式：{result.get('flagos_route', 'FlagGems ATen 训练')}（strict={result['strict']}）",
        f"- 步数：{len(steps)}",
        f"- 总损失：{first:.6f} -> {last:.6f}",
        f"- 初始参数 SHA-256：`{result['initial_parameter_sha256']}`",
        f"- 最终参数 SHA-256：`{result['final_parameter_sha256']}`",
        f"- V100 架构状态：{architecture.get('note', '未知')}",
        f"- CUDA kernel 事件：{result.get('profile', {}).get('kernel_evidence', {}).get('event_count', '未采集')}",
        f"- FlagGems 函数族：{len(result.get('flaggems_log_summary', {}).get('operator_functions', []))}",
        f"- 计算算子 FlagGems kernel 覆盖：{len(execution_summary.get('flaggems_kernel_ops', []))}/{len(execution_summary.get('observed_compute_ops', []))}",
        f"- 原生 kernel fallback：{', '.join(execution_summary.get('native_kernel_ops', [])) or '无'}",
        "",
        "## 训练算子缺口状态",
        "",
        f"- 未批准计算型 fallback：{', '.join(result['operator_inventory']['fallback_compute_ops']) or '无'}",
        "- `cosine_embedding_loss`：已改为等价的基础算子归约。",
        "- AdamW foreach：本轮关闭，使用单张量更新；multi-tensor kernel 留作后续优化。",
        "- AMP/GradScaler：本轮未纳入，FP32 训练通过后单独补齐。",
        "",
        "## 每步",
        "",
        "| Step | Total loss | MCM | PDR | Max grad | Step ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['step']} | {row['loss']:.6f} | {row['mcm_loss']:.6f} | "
        f"{row['pdr_loss']:.6f} | {row['max_grad']:.5g} | {row['step_ms']:.3f} |"
        for row in steps
    )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- checkpoint：`{display_path(result['checkpoint'])}`",
            "- `run.json`：运行状态、版本和逐步指标",
            "- `operator_inventory.json`：训练图算子与 fallback 审计",
            "- `training_profile.json` / `training_trace.json`：可在 Chrome tracing/Perfetto 打开的 profile",
            "- `flaggems_ops.log`：FlagGems 实际注册函数的调试日志",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
