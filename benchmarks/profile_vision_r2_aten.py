"""Capture R2 Vision/KRONOS ATen and kernel traces without timing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from model.flagos_runtime import flagos_inference_scope
from r2_benchmark_common import git_sha, jsonable, runtime_capture, tensor_sha256
from vision_r2_v100_worker import _dtype, _inputs, _invoke, _tolerance


STAGES: Dict[str, Tuple[str, str, bool]] = {
    "marker_torch": ("marker_token_embed", "torch", False),
    "marker_nvidia": ("marker_token_embed", "nvidia", True),
    "swiglu_torch": ("swiglu", "torch", False),
    "swiglu_nvidia": ("swiglu", "nvidia", True),
    "residual_layer_norm_torch": ("residual_layer_norm", "torch", False),
}

ATEN_CLASSIFICATION = {
    "aten::add": "torch_reference_or_input_assembly",
    "aten::addmm": "stock_flagos_aten",
    "aten::bmm": "stock_flagos_aten",
    "aten::baddbmm": "stock_flagos_aten",
    "aten::_softmax": "stock_flagos_aten",
    "aten::embedding": "torch_reference_marker_lookup",
    "aten::masked_fill": "torch_reference_padding",
    "aten::native_layer_norm": "torch_retained_candidate_rejected",
    "aten::silu": "vision_swiglu_candidate",
}


def classify_event(name: str) -> str:
    if name in ATEN_CLASSIFICATION:
        return ATEN_CLASSIFICATION[name]
    if "_marker_token_embed_kernel" in name:
        return "nvidia_custom_marker_token_kernel"
    if "swiglu_kernel" in name:
        return "flaggems_existing_swiglu_kernel_rejected"
    if "layer_norm_kernel" in name:
        return "torch_retained_candidate_rejected"
    return "unclassified"


def _event_value(event: Any, attribute: str) -> float:
    value = getattr(event, attribute, 0.0)
    return float(value) if value is not None else 0.0


def _profile_rows(profile: Any, limit: int) -> List[Dict[str, Any]]:
    rows = []
    for event in profile.key_averages():
        cuda_us = _event_value(event, "self_device_time_total")
        if not cuda_us:
            cuda_us = _event_value(event, "self_cuda_time_total")
        rows.append(
            {
                "name": event.key,
                "count": int(getattr(event, "count", 0)),
                "self_cuda_us": cuda_us,
                "self_cpu_us": _event_value(event, "self_cpu_time_total"),
                "classification": classify_event(event.key),
            }
        )
    rows.sort(key=lambda row: (row["self_cuda_us"], row["self_cpu_us"]), reverse=True)
    return rows[:limit]


def _invoke_and_validate(
    operation: str,
    backend: str,
    tensors: Dict[str, torch.Tensor],
    reference: Tuple[torch.Tensor, Any],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Any, Any]:
    output, dispatch, mask = _invoke(operation, backend, tensors)
    expected, expected_mask = reference
    torch.testing.assert_close(output, expected, **_tolerance(dtype))
    if expected_mask is not None and not torch.equal(mask, expected_mask):
        raise AssertionError("Vision profiling candidate changed the marker padding mask")
    return output, dispatch, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flaggems-source-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--markers", type=int, default=32)
    parser.add_argument("--tokens-per-marker", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--marker-vocab", type=int, default=175)
    parser.add_argument("--swiglu-sequence", type=int, default=264)
    parser.add_argument("--swiglu-hidden", type=int, default=4096)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Vision R2 profiling requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    dtype = _dtype(args.precision)
    operation, backend, use_flagos_scope = STAGES[args.stage]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tensors = _inputs(args, device, dtype)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    with torch.inference_mode():
        reference_value = _invoke(operation, "torch", tensors)[::2]
        if use_flagos_scope:
            with flagos_inference_scope("optimized") as scope_dispatch:
                output, dispatch, _ = _invoke_and_validate(
                    operation, backend, tensors, reference_value, dtype
                )
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=activities, record_shapes=True, profile_memory=True
                ) as profile:
                    _invoke(operation, backend, tensors)
                scope = jsonable(scope_dispatch)
        else:
            output, dispatch, _ = _invoke_and_validate(
                operation, backend, tensors, reference_value, dtype
            )
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True
            ) as profile:
                _invoke(operation, backend, tensors)
            scope = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(args.output_dir / "trace.json"))
    result = {
        "schema_version": 1,
        "stage": args.stage,
        "operation": operation,
        "precision": args.precision,
        "runtime": runtime_capture(device),
        "commits": {"stofm": git_sha(ROOT), "flaggems": git_sha(args.flaggems_source_root)},
        "workload": {
            "batch_size": args.batch_size,
            "markers": args.markers,
            "tokens_per_marker": args.tokens_per_marker,
            "embedding_dim": args.embedding_dim,
            "marker_vocab": args.marker_vocab,
            "swiglu_sequence": args.swiglu_sequence,
            "swiglu_hidden": args.swiglu_hidden,
            "seed": args.seed,
        },
        "dispatch": {"public": jsonable(dispatch), "flagos_scope": scope},
        "validation": {
            "status": "passed",
            "output_sha256": tensor_sha256(output),
            **_tolerance(dtype),
        },
        "events": _profile_rows(profile, args.top),
        "trace": "trace.json",
    }
    (args.output_dir / "profile.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Vision FlagOS Inference R2 ATen Profile",
        "",
        f"Stage: `{args.stage}`; precision: `{args.precision}`.",
        "",
        "This trace is qualitative evidence only; timed latency is reported by the isolated suite.",
        "",
        "| Operator | Calls | Self CUDA us | Self CPU us | Classification |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["events"]:
        lines.append(
            f"| {row['name']} | {row['count']} | {row['self_cuda_us']:.1f} | "
            f"{row['self_cpu_us']:.1f} | {row['classification']} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "event_count": len(result["events"])}, indent=2))


if __name__ == "__main__":
    main()
