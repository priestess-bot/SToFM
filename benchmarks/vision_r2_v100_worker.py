"""One isolated R2 V100 benchmark process for Uni2/KRONOS operator boundaries."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Tuple

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from flag_gems.experimental_ops import marker_token_embed, vit_residual_layer_norm, vit_swiglu
from model.flagos_runtime import flagos_inference_scope
from r2_benchmark_common import benchmark_cuda, git_sha, jsonable, runtime_capture, tensor_sha256


@dataclass(frozen=True)
class StageSpec:
    name: str
    comparison_baseline: str
    operation: str
    backend: str
    use_flagos_scope: bool
    gain_kind: str
    description: str
    status: str = "measured"
    reason: str = ""


STAGES = (
    StageSpec(
        "V0_marker_token_torch",
        "V0_marker_token_torch",
        "marker_token_embed",
        "torch",
        False,
        "reference",
        "KRONOS marker-aware token assembly through the public Torch reference API.",
    ),
    StageSpec(
        "V1_marker_token_nvidia",
        "V0_marker_token_torch",
        "marker_token_embed",
        "nvidia",
        True,
        "custom_kernel",
        "FlagGems NVIDIA marker-token Triton candidate under a real scoped FlagOS inference context.",
    ),
    StageSpec(
        "V2_swiglu_torch",
        "V2_swiglu_torch",
        "swiglu",
        "torch",
        False,
        "reference",
        "Uni2 packed SwiGLU through the public Torch reference API.",
    ),
    StageSpec(
        "V3_swiglu_nvidia",
        "V2_swiglu_torch",
        "swiglu",
        "nvidia",
        True,
        "existing_flaggems_kernel",
        "Existing FlagGems SwiGLU candidate under a real scoped FlagOS inference context.",
    ),
    StageSpec(
        "V4_residual_layer_norm_torch",
        "V4_residual_layer_norm_torch",
        "residual_layer_norm",
        "torch",
        False,
        "reference",
        "Uni2 residual-LayerNorm reference boundary.",
    ),
    StageSpec(
        "V5_residual_layer_norm_rejected",
        "V4_residual_layer_norm_torch",
        "residual_layer_norm",
        "nvidia",
        False,
        "rejected",
        "No NVIDIA residual-LayerNorm native candidate is timed in R2.",
        status="rejected",
        reason="The existing FlagGems skip-LayerNorm candidate lost on the V100 R1 shape and has no verified backward contract.",
    ),
)


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "fp32" else torch.float16


def _tolerance(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float16:
        return {"rtol": 2e-2, "atol": 2e-3}
    return {"rtol": 3e-4, "atol": 3e-5}


def _inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    marker_ids = torch.arange(args.markers, device=device, dtype=torch.long).remainder(args.marker_vocab)
    marker_ids = marker_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    marker_padding = torch.zeros(args.batch_size, args.markers, dtype=torch.bool, device=device)
    padding_count = args.markers // 4
    if padding_count:
        marker_padding[:, -padding_count:] = True
    return {
        "patch_tokens": torch.randn(
            args.batch_size,
            args.markers,
            args.tokens_per_marker,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "marker_ids": marker_ids.masked_fill(marker_padding, -1),
        "marker_padding": marker_padding,
        "marker_weight": torch.randn(args.marker_vocab, args.embedding_dim, device=device, dtype=dtype),
        "position": torch.randn(args.tokens_per_marker, args.embedding_dim, device=device, dtype=dtype),
        "token": torch.randn(args.tokens_per_marker, args.embedding_dim, device=device, dtype=dtype),
        "packed_swiglu": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            2 * args.swiglu_hidden,
            device=device,
            dtype=dtype,
        ),
        "residual_input": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "residual": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "norm_weight": torch.randn(args.embedding_dim, device=device, dtype=dtype),
        "norm_bias": torch.randn(args.embedding_dim, device=device, dtype=dtype),
    }


def _invoke(
    operation: str,
    backend: str,
    tensors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Any, Any]:
    if operation == "marker_token_embed":
        (output, mask), dispatch = marker_token_embed(
            tensors["patch_tokens"],
            tensors["marker_ids"],
            tensors["marker_weight"],
            position_embedding=tensors["position"],
            token_embedding=tensors["token"],
            marker_padding_mask=tensors["marker_padding"],
            backend=backend,
            return_dispatch=True,
        )
        return output, dispatch, mask
    if operation == "swiglu":
        output, dispatch = vit_swiglu(tensors["packed_swiglu"], backend=backend, return_dispatch=True)
        return output, dispatch, None
    if operation == "residual_layer_norm":
        output, dispatch = vit_residual_layer_norm(
            tensors["residual_input"],
            tensors["residual"],
            (tensors["residual_input"].shape[-1],),
            tensors["norm_weight"],
            tensors["norm_bias"],
            backend=backend,
            return_dispatch=True,
        )
        return output, dispatch, None
    raise ValueError(f"unsupported operation: {operation}")


def _run_stage(
    spec: StageSpec,
    args: argparse.Namespace,
    tensors: Dict[str, torch.Tensor],
    references: Dict[str, Tuple[torch.Tensor, Any]],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    if spec.status != "measured":
        return {
            "stage": spec.name,
            "scope": "vision_operator",
            "status": spec.status,
            "comparison_baseline": spec.comparison_baseline,
            "gain_kind": spec.gain_kind,
            "description": spec.description,
            "reason": spec.reason,
            "samples_ms": [],
        }

    expected, expected_mask = references[spec.operation]
    tolerance = _tolerance(dtype)

    def invoke():
        return _invoke(spec.operation, spec.backend, tensors)

    if spec.use_flagos_scope:
        with flagos_inference_scope("optimized") as scope_dispatch:
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
    else:
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
        runtime = None

    return {
        "stage": spec.name,
        "scope": "vision_operator",
        "status": "measured",
        "comparison_baseline": spec.comparison_baseline,
        "gain_kind": spec.gain_kind,
        "description": spec.description,
        "validation": {"status": "passed", "output_sha256": tensor_sha256(output), **tolerance},
        "dispatch": {"public": jsonable(dispatch), "flagos_scope": runtime},
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
            operation: _invoke(operation, "torch", tensors)[::2]
            for operation in ("marker_token_embed", "swiglu", "residual_layer_norm")
        }
        results = [_run_stage(spec, args, tensors, references, dtype) for spec in STAGES]

    result = {
        "schema_version": 1,
        "run_id": f"vision-r2-{args.precision}-{args.run_index:02d}",
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
