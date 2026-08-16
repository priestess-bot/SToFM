#!/usr/bin/env python3
"""Measure SToFM MUSA operator boundaries across a reproducible shape matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from stofm_musa_s4000_worker import _measure, _sha256_file, _tolerance


ROOT = Path(__file__).resolve().parents[1]


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _verified_revision(path: Path, expected: Optional[str], label: str) -> str:
    actual = _git_revision(path)
    if expected and actual != expected:
        raise RuntimeError(f"{label} checkout is {actual}, expected {expected}")
    return expected or actual


def _parse_csv(values: str, converter) -> List[Any]:
    return [converter(value.strip()) for value in values.split(",") if value.strip()]


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float(
            (difference / expected.float().abs().clamp_min(1e-12)).max().item()
        ),
    }


def _gaussian_inputs(
    *,
    nodes: int,
    hidden: int,
    heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
    distances = torch.rand(1, nodes, nodes, device=device, dtype=dtype)
    distances[:, 0, 0] = 0.0
    parameters = (
        torch.ones(1, 1, device=device, dtype=dtype),
        torch.zeros(1, device=device, dtype=dtype),
        torch.rand(1, hidden, device=device, dtype=dtype).mul_(3.0),
        torch.rand(1, hidden, device=device, dtype=dtype).mul_(3.0),
        torch.randn(hidden, hidden, device=device, dtype=dtype).mul_(0.02),
        torch.zeros(hidden, device=device, dtype=dtype),
        torch.randn(heads, hidden, device=device, dtype=dtype).mul_(0.02),
        torch.zeros(heads, device=device, dtype=dtype),
    )
    return distances, parameters, distances.eq(0.0)


def _pair_inputs(
    *,
    nodes: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(1, heads, nodes, head_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    pair_bias = torch.randn(1, heads, nodes, nodes, device=device, dtype=dtype)
    return query, key, value, pair_bias


def _pair_cpu_oracle(
    pair_attention_reference,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair_bias: torch.Tensor,
):
    return pair_attention_reference(
        query.cpu().float(),
        key.cpu().float(),
        value.cpu().float(),
        pair_bias.cpu().float(),
        return_pair=False,
        return_weights=False,
    )[0]


def _validate_gaussian(
    gaussian_pair_bias_dense,
    candidate,
    distances: torch.Tensor,
    parameters: Tuple[torch.Tensor, ...],
    zero_mask: torch.Tensor,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    with torch.inference_mode():
        expected = gaussian_pair_bias_dense(distances, *parameters, zero_mask)
        actual = candidate(distances, parameters, zero_mask)
        torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, **_tolerance(dtype))
    return {"oracle": "MUSA portable PyTorch expression", "errors": _max_error(actual, expected)}


def _validate_pair(
    pair_attention_reference,
    candidate,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair_bias: torch.Tensor,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    with torch.inference_mode():
        actual = candidate(
            query, key, value, pair_bias, query.shape[-1] ** -0.5
        )
        if dtype == torch.bfloat16:
            expected = _pair_cpu_oracle(pair_attention_reference, query, key, value, pair_bias)
            torch.testing.assert_close(actual.float().cpu(), expected, rtol=1e-2, atol=1e-2)
            errors = _max_error(actual.float().cpu(), expected)
            oracle = "CPU FP32 expression on the same BF16 input values"
        else:
            expected = pair_attention_reference(
                query,
                key,
                value,
                pair_bias,
                return_pair=False,
                return_weights=False,
            )[0]
            torch.testing.assert_close(actual, expected, **_tolerance(dtype))
            errors = _max_error(actual, expected)
            oracle = "MUSA portable PyTorch expression"
        torch.musa.synchronize()
    return {"oracle": oracle, "errors": errors}


def _speedup(reference: Dict[str, Any], native: Dict[str, Any]) -> float:
    return reference["p50_ms"] / native["p50_ms"]


def _measure_inference(fn, **kwargs) -> Dict[str, Any]:
    with torch.inference_mode():
        return _measure(fn, **kwargs)


def _write(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["operator", "precision", "nodes", "implementation", "sample_index", "device_event_ms", "host_wall_ms"],
        )
        writer.writeheader()
        for row in result["results"]:
            for implementation, label in (
                ("torch_reference", "PyTorch reference"),
                ("flagos_candidate", row["candidate_label"]),
            ):
                measurements = row[implementation]
                for index, event_ms in enumerate(measurements["samples_ms"]):
                    writer.writerow(
                        {
                            "operator": row["operator"],
                            "precision": row["precision"],
                            "nodes": row["nodes"],
                            "implementation": label,
                            "sample_index": index,
                            "device_event_ms": event_ms,
                            "host_wall_ms": measurements["host_samples_ms"][index],
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flaggems-root", type=Path, default=ROOT.parent / "FlagGems-stofm")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--stofm-revision")
    parser.add_argument("--flaggems-revision")
    parser.add_argument(
        "--candidate",
        choices=("direct-privateuse1", "flagos-backend"),
        default="flagos-backend",
        help="Measure either the frozen direct kernel or the optimized public FlagOS backend.",
    )
    parser.add_argument("--shapes", default="256,512,1050,2048")
    parser.add_argument("--precisions", default="fp32,fp16,bf16")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    shapes = _parse_csv(args.shapes, int)
    precisions = _parse_csv(args.precisions, str)
    if not shapes or not precisions or any(nodes <= 0 for nodes in shapes):
        raise ValueError("--shapes and --precisions must be non-empty positive values")
    if args.gaussian_hidden_dim > 128 or args.heads > 16:
        raise ValueError("the current native Gaussian kernel supports hidden_dim <= 128 and heads <= 16")
    library = args.library.resolve()
    if not library.is_file():
        raise FileNotFoundError(f"MUSA extension library does not exist: {library}")
    source_root = args.flaggems_root.resolve() / "src"
    if not (source_root / "flag_gems").is_dir():
        raise FileNotFoundError(f"FlagGems source package does not exist: {source_root}")
    stofm_revision = _verified_revision(ROOT, args.stofm_revision, "SToFM")
    flaggems_revision = _verified_revision(
        args.flaggems_root, args.flaggems_revision, "FlagGems"
    )
    sys.path.insert(0, str(source_root))
    os.environ["FLAGGEMS_STOFM_MUSA_LIBRARY"] = str(library)
    os.environ["FLAGGEMS_STOFM_ENABLE_MUSA_NATIVE"] = "1"
    os.environ["FLAGGEMS_STOFM_REQUIRE_MUSA_NATIVE"] = "1"
    os.environ["FLAGGEMS_STOFM_MUSA_MINIMAL_IMPORT"] = "1"

    import torch_musa  # noqa: F401
    from flag_gems.experimental_ops._stofm_common import gaussian_pair_bias_dense, pair_attention_reference
    from flag_gems.experimental_ops.stofm_backends import mthreads

    if not torch.musa.is_available():
        raise RuntimeError("torch.musa.is_available() is false")
    torch.ops.load_library(str(library))
    direct_gaussian = torch.ops.flagos_stofm.gaussian_pair_bias
    direct_pair = torch.ops.flagos_stofm.pair_score_epilogue
    if args.candidate == "flagos-backend":
        candidate_label = "Optimized FlagOS MUSA backend"

        def gaussian_candidate(distances, parameters, zero_mask):
            return mthreads.gaussian_pair_bias(distances, *parameters, zero_mask)

        def pair_candidate(query, key, value, pair_bias, scale):
            return mthreads.pair_attention(
                query,
                key,
                value,
                pair_bias,
                key_padding_mask=None,
                dropout_p=0.0,
                training=False,
                scale=scale,
                return_pair=False,
                return_weights=False,
                assume_finite_pair_bias=True,
            )[0]

    else:
        candidate_label = "Initial FlagOS MUSA registered implementation"

        def gaussian_candidate(distances, parameters, zero_mask):
            return direct_gaussian(distances, *parameters, zero_mask)

        def pair_candidate(query, key, value, pair_bias, scale):
            return direct_pair(
                query, key, value, pair_bias, None, scale, False, False
            )[0]

    runtime = torch.musa
    device = torch.device("musa:0")
    results = []
    for precision_index, precision in enumerate(precisions):
        dtype = _dtype(precision)
        for shape_index, nodes in enumerate(shapes):
            torch.manual_seed(args.seed + precision_index * 1000 + shape_index)
            distances, gaussian_parameters, zero_mask = _gaussian_inputs(
                nodes=nodes,
                hidden=args.gaussian_hidden_dim,
                heads=args.heads,
                dtype=dtype,
                device=device,
            )
            gaussian_validation = _validate_gaussian(
                gaussian_pair_bias_dense,
                gaussian_candidate,
                distances,
                gaussian_parameters,
                zero_mask,
                dtype,
            )
            gaussian_reference = _measure_inference(
                lambda distances=distances, gaussian_parameters=gaussian_parameters, zero_mask=zero_mask: gaussian_pair_bias_dense(
                    distances, *gaussian_parameters, zero_mask
                ),
                runtime=runtime,
                warmup=args.warmup,
                repetitions=args.repetitions,
                calls_per_sample=args.calls_per_sample,
                seed=args.seed + shape_index,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            gaussian_native = _measure_inference(
                lambda distances=distances, gaussian_parameters=gaussian_parameters, zero_mask=zero_mask: gaussian_candidate(
                    distances, gaussian_parameters, zero_mask
                ),
                runtime=runtime,
                warmup=args.warmup,
                repetitions=args.repetitions,
                calls_per_sample=args.calls_per_sample,
                seed=args.seed + 100 + shape_index,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            results.append(
                {
                    "operator": "gaussian_pair_bias",
                    "precision": precision,
                    "nodes": nodes,
                    "candidate_label": candidate_label,
                    "validation": gaussian_validation,
                    "torch_reference": gaussian_reference,
                    "flagos_candidate": gaussian_native,
                    "p50_speedup": _speedup(gaussian_reference, gaussian_native),
                }
            )
            del distances, gaussian_parameters, zero_mask
            runtime.empty_cache()

            query, key, value, pair_bias = _pair_inputs(
                nodes=nodes,
                heads=args.heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
            )
            pair_validation = _validate_pair(
                pair_attention_reference,
                pair_candidate,
                query,
                key,
                value,
                pair_bias,
                dtype,
            )
            scale = args.head_dim**-0.5
            pair_reference = _measure_inference(
                lambda query=query, key=key, value=value, pair_bias=pair_bias: pair_attention_reference(
                    query, key, value, pair_bias, return_pair=False, return_weights=False
                ),
                runtime=runtime,
                warmup=args.warmup,
                repetitions=args.repetitions,
                calls_per_sample=args.calls_per_sample,
                seed=args.seed + 200 + shape_index,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            pair_native = _measure_inference(
                lambda query=query, key=key, value=value, pair_bias=pair_bias, scale=scale: pair_candidate(
                    query, key, value, pair_bias, scale
                ),
                runtime=runtime,
                warmup=args.warmup,
                repetitions=args.repetitions,
                calls_per_sample=args.calls_per_sample,
                seed=args.seed + 300 + shape_index,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            results.append(
                {
                    "operator": "pair_score_epilogue",
                    "precision": precision,
                    "nodes": nodes,
                    "candidate_label": candidate_label,
                    "validation": pair_validation,
                    "torch_reference": pair_reference,
                    "flagos_candidate": pair_native,
                    "p50_speedup": _speedup(pair_reference, pair_native),
                }
            )
            del query, key, value, pair_bias
            runtime.empty_cache()

    properties = runtime.get_device_properties(device)
    result = {
        "schema_version": 1,
        "role": "musa_operator_matrix",
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("musa-operator-matrix-%Y%m%dT%H%M%SZ"),
        "trial": args.trial,
        "candidate": {
            "kind": args.candidate,
            "label": candidate_label,
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_musa": torch_musa.__version__,
            "device": runtime.get_device_name(device),
            "capability": list(runtime.get_device_capability(device)),
            "total_memory_mib": round(float(properties.total_memory) / 1024**2, 2),
        },
        "sources": {
            "stofm_revision": stofm_revision,
            "flaggems_revision": flaggems_revision,
            "musa_library_sha256": _sha256_file(library),
        },
        "workload": {
            "shapes": shapes,
            "precisions": precisions,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "seed": args.seed,
        },
        "measurement": {
            "timer": "torch.musa.Event with per-sample device synchronization",
            "host_cross_check": "time.perf_counter",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "calls_per_sample": args.calls_per_sample,
            "compile_included": False,
            "inference_mode": True,
        },
        "results": results,
    }
    _write(args.output_dir, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
