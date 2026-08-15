"""Measure Vision/KRONOS boundaries in the immutable frozen-FlagOS environment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from model.flagos_runtime import flagos_inference_scope
from r2_benchmark_common import benchmark_cuda, git_sha, jsonable, runtime_capture, tensor_sha256
from vision_r2_common import _dtype, _inputs, _tolerance, torch_invoke


@dataclass(frozen=True)
class StockStageSpec:
    name: str
    operation: str
    description: str


@dataclass(frozen=True)
class StockVisionDispatch:
    """Frozen-environment record shaped like the R2 public Vision dispatch."""

    operator: str
    requested: str
    selected: str
    precision: str
    reason: str


STOCK_STAGES = (
    StockStageSpec(
        "V0s_marker_token_stock_flagos",
        "marker_token_embed",
        "Frozen FlagOS scoped ATen baseline for KRONOS marker-token assembly.",
    ),
    StockStageSpec(
        "V2s_swiglu_stock_flagos",
        "swiglu",
        "Frozen FlagOS scoped ATen baseline for Uni2 packed SwiGLU.",
    ),
    StockStageSpec(
        "V4s_residual_layer_norm_stock_flagos",
        "residual_layer_norm",
        "Frozen FlagOS scoped ATen baseline for Uni2 residual LayerNorm.",
    ),
)


def _precision_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float32:
        return "fp32"
    if tensor.dtype == torch.float16:
        return "fp16"
    if tensor.dtype == torch.bfloat16:
        return "bf16"
    return str(tensor.dtype).replace("torch.", "")


def _validate_marker_token_inputs(tensors: Dict[str, torch.Tensor]) -> None:
    """Mirror the versioned R2 public API validation without importing it.

    The frozen checkout predates the Vision API.  Keeping its reference wrapper
    equivalent to the Torch/optimized public boundary prevents host-side
    validation and dispatch construction from biasing the F0 comparison.
    """
    patch_tokens = tensors["patch_tokens"]
    marker_ids = tensors["marker_ids"]
    marker_weight = tensors["marker_weight"]
    position = tensors["position"]
    token = tensors["token"]
    padding = tensors["marker_padding"]
    if patch_tokens.ndim != 4:
        raise ValueError("patch_tokens must have shape [batch, markers, tokens, embedding_dim]")
    batch, markers, tokens, embedding_dim = patch_tokens.shape
    if marker_ids.shape != (batch, markers) or marker_ids.dtype != torch.long:
        raise ValueError("marker_ids must be a torch.long tensor with shape [batch, markers]")
    if marker_weight.ndim != 2 or marker_weight.shape[1] != embedding_dim:
        raise ValueError("marker_embedding_weight must have shape [num_markers, embedding_dim]")
    if position.shape != (tokens, embedding_dim):
        raise ValueError("position_embedding must have shape [tokens, embedding_dim]")
    if token.shape not in {(embedding_dim,), (tokens, embedding_dim)}:
        raise ValueError("token_embedding must have shape [embedding_dim] or [tokens, embedding_dim]")
    if padding.shape != (batch, markers):
        raise ValueError("marker_padding_mask must have shape [batch, markers]")
    values = (marker_ids, marker_weight, position, token, padding)
    if any(value.device != patch_tokens.device for value in values):
        raise ValueError("all marker-token tensors must share patch_tokens.device")


def _validate_residual_layer_norm_inputs(tensors: Dict[str, torch.Tensor]) -> None:
    input_tensor = tensors["residual_input"]
    if input_tensor.shape != tensors["residual"].shape:
        raise ValueError("input_tensor and residual must have the same shape")
    if tuple(input_tensor.shape[-1:]) != (input_tensor.shape[-1],):
        raise ValueError("normalized_shape does not match input_tensor")


def _invoke(operation: str, tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, StockVisionDispatch, Any]:
    input_tensor = tensors["patch_tokens"] if operation == "marker_token_embed" else tensors["residual_input"]
    if operation == "swiglu":
        input_tensor = tensors["packed_swiglu"]
        if input_tensor.shape[-1] % 2:
            raise ValueError("packed SwiGLU input must have an even final dimension")
    elif operation == "marker_token_embed":
        _validate_marker_token_inputs(tensors)
    elif operation == "residual_layer_norm":
        _validate_residual_layer_norm_inputs(tensors)
    else:
        raise ValueError(f"unsupported operation: {operation}")
    output, mask = torch_invoke(operation, tensors)
    dispatch = StockVisionDispatch(
        operator=operation,
        requested="stock",
        selected="torch",
        precision=_precision_name(input_tensor),
        reason="frozen FlagOS has no Vision composite API; reference boundary executes inside scoped ATen dispatch",
    )
    return output, dispatch, mask


def _run_stage(
    spec: StockStageSpec,
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
    references: Dict[str, Tuple[torch.Tensor, Any]],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    expected, expected_mask = references[spec.operation]
    tolerance = _tolerance(dtype)

    def invoke():
        return _invoke(spec.operation, tensors)

    with flagos_inference_scope("stock") as scope_dispatch:
        output, dispatch, mask = invoke()
        torch.testing.assert_close(output, expected, **tolerance)
        if expected_mask is not None and not torch.equal(mask, expected_mask):
            raise AssertionError(f"{spec.name} changed the marker padding mask")
        measured = benchmark_cuda(
            invoke,
            warmup=args.warmup,
            repetitions=args.repetitions,
            calls_per_sample=args.calls_per_sample,
        )
        runtime = jsonable(scope_dispatch)

    return {
        "stage": spec.name,
        "scope": "vision_operator",
        "status": "measured",
        "comparison_baseline": spec.name,
        "gain_kind": "stock_aten_reference",
        "description": spec.description,
        "validation": {"status": "passed", "output_sha256": tensor_sha256(output), **tolerance},
        "dispatch": {
            "public": jsonable(dispatch),
            "flagos_scope": runtime,
        },
        **measured,
    }


def _write_result(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "sample_index", "latency_ms"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in result["results"]:
            for index, sample in enumerate(row["samples_ms"]):
                writer.writerow({"stage": row["stage"], "sample_index": index, "latency_ms": sample})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--flaggems-source-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--markers", type=int, default=32)
    parser.add_argument("--tokens-per-marker", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--marker-vocab", type=int, default=175)
    parser.add_argument("--swiglu-sequence", type=int, default=264)
    parser.add_argument("--swiglu-hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Vision R2 V100 benchmark requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    dtype = _dtype(args.precision)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tensors = _inputs(args, device, dtype)

    with torch.inference_mode():
        references = {
            operation: _invoke(operation, tensors)[::2]
            for operation in ("marker_token_embed", "swiglu", "residual_layer_norm")
        }
        results = [_run_stage(spec, args, tensors, references, dtype) for spec in STOCK_STAGES]

    result = {
        "schema_version": 2,
        "role": "stock",
        "run_id": f"vision-r2-stock-{args.precision}-{args.run_index:02d}",
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
        "measurement": {
            "timer": "cuda_events",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "calls_per_sample": args.calls_per_sample,
            "compile_included": False,
            "tf32": False,
            "inference_mode": True,
        },
        "reference_hashes": {operation: tensor_sha256(value[0]) for operation, value in references.items()},
        "results": results,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_result(args.output_dir, result)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
