"""One isolated R2 SToFM V100 benchmark process.

Run this worker once with the frozen-stock FlagGems environment and once with
the optimized FlagGems environment. ``run_stofm_r2_v100_suite.py`` coordinates
the independent processes and aggregates their raw samples.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.se2transformer import SToFMModel
from model.utils import SToFMConfig
from r2_benchmark_common import (
    benchmark_cuda,
    git_sha,
    jsonable,
    runtime_capture,
    tensor_sha256,
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    mode: str
    gaussian_backend: str
    attention_backend: str
    return_pair_rep: bool
    measurement_scope: str
    comparison_baseline: str
    gain_kind: str
    description: str


def _stage_specs(role: str) -> List[StageSpec]:
    if role == "stock":
        return [
            StageSpec(
                "P0_legacy_pair_output",
                "torch",
                "torch",
                "torch",
                True,
                "none",
                "P0_legacy_pair_output",
                "model_lifecycle",
                "Legacy end-to-end Torch inference that materializes the final pair representation.",
            ),
            StageSpec(
                "P1_canonical_torch",
                "torch",
                "torch",
                "torch",
                False,
                "none",
                "P1_canonical_torch",
                "model_lifecycle",
                "Canonical Torch inference that omits the unused final pair representation.",
            ),
            StageSpec(
                "F0_stock_lifecycle",
                "stock",
                "torch",
                "torch",
                False,
                "per_call",
                "P1_canonical_torch",
                "flaggems_lifecycle",
                "Frozen FlagGems stock ATen scope created and destroyed for every model call.",
            ),
            StageSpec(
                "F0_stock_steady",
                "stock",
                "torch",
                "torch",
                False,
                "steady",
                "P1_canonical_torch",
                "stock_aten",
                "Frozen FlagGems stock ATen scope kept open across timed calls.",
            ),
        ]
    if role == "optimized":
        return [
            StageSpec(
                "C1_gaussian_compiler",
                "optimized",
                "inductor",
                "torch",
                False,
                "steady",
                "F0_stock_steady",
                "compiler_routing",
                "FlagGems ATen plus the explicit SToFM Gaussian compiler candidate.",
            ),
            StageSpec(
                "C2_pair_native_epilogue",
                "optimized",
                "torch",
                "nvidia",
                False,
                "steady",
                "F0_stock_steady",
                "custom_kernel",
                "FlagGems ATen plus the native SToFM pair-score epilogue candidate.",
            ),
            StageSpec(
                "Ffinal_optimized_lifecycle",
                "optimized",
                "flaggems",
                "flaggems",
                False,
                "per_call",
                "F0_stock_steady",
                "flaggems_lifecycle",
                "Combined optimized FlagGems route including temporary scope lifecycle.",
            ),
            StageSpec(
                "Ffinal_optimized_steady",
                "optimized",
                "flaggems",
                "flaggems",
                False,
                "steady",
                "F0_stock_steady",
                "combined",
                "Combined FlagGems ATen, Gaussian compiler, and pair epilogue steady-state route.",
            ),
        ]
    raise ValueError(f"unsupported worker role: {role}")


def _config(spec: StageSpec, args: argparse.Namespace) -> SToFMConfig:
    return SToFMConfig(
        num_hidden_layers=args.layers,
        embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.ffn_embedding_dim,
        num_attention_heads=args.heads,
        gaussian_hidden_dim=args.gaussian_hidden_dim,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        input_dim=args.input_dim,
        flagos_mode=spec.mode,
        flagos_backend=spec.gaussian_backend,
        flagos_attention_backend=spec.attention_backend,
    )


def _make_model(
    spec: StageSpec,
    args: argparse.Namespace,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> SToFMModel:
    model = SToFMModel(_config(spec, args)).to(device).eval()
    if state_dict:
        model.load_state_dict(state_dict)
    if dtype == torch.float16:
        model.half()
    return model


def _tolerance(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float16:
        return {"rtol": 3e-2, "atol": 3e-3}
    return {"rtol": 3e-4, "atol": 3e-5}


def _dispatch_snapshot(model: SToFMModel) -> Dict[str, Any]:
    first_attention = model.encoder.layers[0].self_attn
    return jsonable(
        {
            "runtime": model.last_flagos_runtime_dispatch,
            "gaussian": model.gaussian.last_flagos_dispatch,
            "pair_attention_layer0": first_attention.last_flagos_dispatch,
        }
    )


def _run_stage(
    spec: StageSpec,
    args: argparse.Namespace,
    state_dict: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
    expected: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    model = _make_model(spec, args, state_dict, device, dtype)

    def invoke():
        return model(
            inputs["token_embeddings"],
            inputs["attn_bias"],
            inputs["token_types"],
            return_pair_rep=spec.return_pair_rep,
        )

    output = invoke()
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(output["last_hidden_state"], expected, **tolerance)
    if spec.return_pair_rep:
        if "pair_rep" not in output:
            raise AssertionError(f"{spec.name} must materialize pair_rep")
    elif "pair_rep" in output:
        raise AssertionError(f"{spec.name} must omit the final pair_rep")

    if spec.measurement_scope == "steady" and spec.mode != "torch":
        with model.flagos_inference_scope() as scope_dispatch:
            measured = benchmark_cuda(
                invoke,
                warmup=args.warmup,
                repetitions=args.repetitions,
                calls_per_sample=args.calls_per_sample,
            )
            dispatch = _dispatch_snapshot(model)
            dispatch["outer_scope"] = jsonable(scope_dispatch)
    else:
        measured = benchmark_cuda(
            invoke,
            warmup=args.warmup,
            repetitions=args.repetitions,
            calls_per_sample=args.calls_per_sample,
        )
        dispatch = _dispatch_snapshot(model)

    return {
        "stage": spec.name,
        "scope": "end_to_end",
        "status": "measured",
        "comparison_baseline": spec.comparison_baseline,
        "gain_kind": spec.gain_kind,
        "measurement_scope": spec.measurement_scope,
        "description": spec.description,
        "validation": {
            "status": "passed",
            "last_hidden_state_sha256": tensor_sha256(output["last_hidden_state"]),
            **tolerance,
        },
        "dispatch": dispatch,
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
        )
        writer.writeheader()
        for row in result["results"]:
            for index, sample in enumerate(row["samples_ms"]):
                writer.writerow(
                    {
                        "stage": row["stage"],
                        "sample_index": index,
                        "latency_ms": sample,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["stock", "optimized"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ffn-embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--flaggems-source-root", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SToFM R2 V100 benchmark requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    baseline_spec = StageSpec(
        "reference",
        "torch",
        "torch",
        "torch",
        False,
        "none",
        "reference",
        "reference",
        "Canonical Torch reference.",
    )
    baseline = _make_model(baseline_spec, args, {}, device, dtype)
    state_dict = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    inputs = {
        "token_embeddings": torch.randn(
            args.batch_size, args.nodes, args.input_dim, device=device, dtype=dtype
        ),
        "attn_bias": torch.rand(
            args.batch_size, args.nodes, args.nodes, device=device, dtype=dtype
        ),
        "token_types": torch.zeros(
            args.batch_size, args.nodes, dtype=torch.long, device=device
        ),
    }
    inputs["attn_bias"][:, 0, 0] = 0.0

    with torch.inference_mode():
        reference_output = baseline(
            inputs["token_embeddings"],
            inputs["attn_bias"],
            inputs["token_types"],
            return_pair_rep=False,
        )
        expected = reference_output["last_hidden_state"].detach().clone()
        results = [
            _run_stage(spec, args, state_dict, inputs, expected, device, dtype)
            for spec in _stage_specs(args.role)
        ]

    result = {
        "schema_version": 2,
        "run_id": f"stofm-r2-{args.precision}-{args.role}-{args.run_index:02d}",
        "role": args.role,
        "precision": args.precision,
        "runtime": runtime_capture(device),
        "commits": {
            "stofm": git_sha(ROOT),
            "flaggems": git_sha(args.flaggems_source_root),
        },
        "workload": {
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "layers": args.layers,
            "embedding_dim": args.embedding_dim,
            "ffn_embedding_dim": args.ffn_embedding_dim,
            "heads": args.heads,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "input_dim": args.input_dim,
            "seed": args.seed,
            "return_pair_rep": False,
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
        "reference": {
            "stage": "P1_canonical_torch",
            "last_hidden_state_sha256": tensor_sha256(expected),
            **_tolerance(dtype),
        },
        "results": results,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _write_result(args.output_dir, result)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
