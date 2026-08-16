#!/usr/bin/env python3
"""Run one correctness-gated SToFM native-MUSA benchmark trial on MTT S4000.

This worker measures only the MUSA extension-backed SToFM boundaries.  It does
not claim that generic FlagGems ATen/Triton substitutions are active: MUSA 3.1
has no Triton driver, and that unavailable route is emitted as an explicit
result row.  Run it in a fresh Python process for every independent trial.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StageSpec:
    identifier: str
    description: str
    gaussian_backend: str
    attention_backend: str
    baseline_identifier: str
    expected_operators: Tuple[str, ...] = ()


STAGES: Tuple[StageSpec, ...] = (
    StageSpec(
        "pure_pytorch",
        "Pure PyTorch inference with no FlagOS import or operator replacement.",
        "torch",
        "torch",
        "pure_pytorch",
    ),
    StageSpec(
        "native_gaussian_pair_bias",
        "Only the PrivateUse1 Gaussian pair-bias operator; pair-score stays in PyTorch.",
        "mthreads",
        "torch",
        "pure_pytorch",
        ("flagos_stofm::gaussian_pair_bias",),
    ),
    StageSpec(
        "native_pair_score_epilogue",
        "Only the PrivateUse1 pair-score softmax/context operator; Gaussian stays in PyTorch.",
        "torch",
        "mthreads",
        "pure_pytorch",
        ("flagos_stofm::pair_score_epilogue",),
    ),
    StageSpec(
        "native_stofm_operators_combined",
        "Both PrivateUse1 SToFM operators with global FlagGems ATen dispatch disabled.",
        "mthreads",
        "mthreads",
        "pure_pytorch",
        (
            "flagos_stofm::gaussian_pair_bias",
            "flagos_stofm::pair_score_epilogue",
        ),
    ),
)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _quantile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summarize_samples(samples: Sequence[float], *, seed: int, resamples: int) -> Dict[str, Any]:
    if not samples:
        return {}
    generator = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [samples[generator.randrange(len(samples))] for _ in samples]
        medians.append(statistics.median(sample))
    mean = statistics.mean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "p20_ms": _quantile(samples, 0.20),
        "p50_ms": _quantile(samples, 0.50),
        "p80_ms": _quantile(samples, 0.80),
        "p95_ms": _quantile(samples, 0.95),
        "mean_ms": mean,
        "stdev_ms": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
        "bootstrap_p50_ci_ms": [_quantile(medians, 0.025), _quantile(medians, 0.975)],
        "bootstrap_resamples": resamples,
    }


def _synchronize(runtime: Any) -> None:
    runtime.synchronize()


def _memory_mib(runtime: Any, name: str) -> Optional[float]:
    function = getattr(runtime, name, None)
    if function is None:
        return None
    try:
        return float(function()) / 1024**2
    except (RuntimeError, TypeError):
        return None


def _measure(
    fn: Callable[[], object],
    *,
    runtime: Any,
    warmup: int,
    repetitions: int,
    calls_per_sample: int,
    seed: int,
    bootstrap_resamples: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    _synchronize(runtime)
    runtime.reset_peak_memory_stats()
    allocated_before = _memory_mib(runtime, "memory_allocated")
    event_samples: List[float] = []
    host_samples: List[float] = []
    for _ in range(repetitions):
        _synchronize(runtime)
        start = runtime.Event(enable_timing=True)
        end = runtime.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        for _ in range(calls_per_sample):
            fn()
        end.record()
        end.synchronize()
        host_elapsed = (time.perf_counter() - host_start) * 1000.0 / calls_per_sample
        event_samples.append(start.elapsed_time(end) / calls_per_sample)
        host_samples.append(host_elapsed)
    allocated_peak = _memory_mib(runtime, "max_memory_allocated")
    measurement = {
        "sample_count": len(event_samples),
        "calls_per_sample": calls_per_sample,
        "samples_ms": event_samples,
        "host_samples_ms": host_samples,
        "peak_allocated_mib": allocated_peak,
        "peak_delta_allocated_mib": (
            allocated_peak - allocated_before
            if allocated_peak is not None and allocated_before is not None
            else None
        ),
    }
    measurement.update(
        _summarize_samples(event_samples, seed=seed, resamples=bootstrap_resamples)
    )
    measurement["host_timing"] = _summarize_samples(
        host_samples, seed=seed + 1, resamples=bootstrap_resamples
    )
    return measurement


def _measure_inference(fn: Callable[[], object], **kwargs) -> Dict[str, Any]:
    with torch.inference_mode():
        return _measure(fn, **kwargs)


def _dtype(name: str) -> torch.dtype:
    values = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    return values[name]


def _tolerance(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float32:
        return {"rtol": 4e-4, "atol": 4e-5}
    if dtype == torch.float16:
        return {"rtol": 3e-2, "atol": 3e-3}
    return {"rtol": 5e-2, "atol": 1e-2}


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-12)
    return {
        "max_abs_error": float(difference.max().item()),
        "max_relative_error": float((difference / denominator).max().item()),
    }


def _config(args: argparse.Namespace, spec: StageSpec):
    from model.utils import SToFMConfig

    native = spec.gaussian_backend == "mthreads" or spec.attention_backend == "mthreads"
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
        flagos_mode="optimized" if native else "torch",
        flagos_backend=spec.gaussian_backend,
        flagos_attention_backend=spec.attention_backend,
        # The target's generic Triton has no MUSA driver. Native-operator
        # measurement must therefore isolate the two explicit PrivateUse1 ops.
        flagos_aten_dispatch=False,
    )


def _dispatch_snapshot(model: Any) -> Dict[str, Any]:
    def _record(value: Any) -> Optional[Dict[str, str]]:
        if value is None:
            return None
        return {
            "operator": value.operator,
            "requested": value.requested,
            "selected": value.selected,
            "precision": value.precision,
            "reason": value.reason,
        }

    return {
        "runtime": {
            "active": bool(model.last_flagos_runtime_dispatch and model.last_flagos_runtime_dispatch.active),
            "reason": (
                model.last_flagos_runtime_dispatch.reason
                if model.last_flagos_runtime_dispatch is not None
                else "no FlagOS runtime scope was entered"
            ),
        },
        "gaussian": _record(model.gaussian.last_flagos_dispatch),
        "pair_attention_layers": [
            _record(layer.self_attn.last_flagos_dispatch) for layer in model.encoder.layers
        ],
    }


def _validate_stage(
    spec: StageSpec,
    model: Any,
    reference: torch.Tensor,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    tokens, distances, token_types = inputs
    with torch.inference_mode():
        output = model(tokens, distances, token_types, return_pair_rep=False)["last_hidden_state"]
        torch.musa.synchronize()
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(output, reference, **tolerance)
    dispatch = _dispatch_snapshot(model)
    selected = []
    if dispatch["gaussian"] is not None:
        selected.append(dispatch["gaussian"]["selected"])
    selected.extend(
        value["selected"] for value in dispatch["pair_attention_layers"] if value is not None
    )
    expected_native_count = int("flagos_stofm::gaussian_pair_bias" in spec.expected_operators)
    expected_native_count += int("flagos_stofm::pair_score_epilogue" in spec.expected_operators) * len(
        model.encoder.layers
    )
    if expected_native_count:
        assert selected.count("mthreads") == expected_native_count, (
            f"{spec.identifier} did not execute the expected MUSA direct operators: {dispatch}"
        )
    return {
        "status": "passed",
        "tolerance": tolerance,
        "errors": _max_errors(output, reference),
        "output_sha256": _tensor_digest(output),
        "dispatch": dispatch,
    }


def _unavailable_aten_row() -> Dict[str, Any]:
    return {
        "stage": "native_stofm_operators_plus_flagos_aten",
        "description": (
            "Requested combination of native SToFM operators and existing FlagGems ATen dispatch."
        ),
        "baseline_stage": "pure_pytorch",
        "status": "unavailable",
        "reason": (
            "The target MUSA 3.1 environment has no Triton driver, so generic FlagGems "
            "ATen substitutions cannot execute. This row intentionally has no substituted timing."
        ),
        "samples_ms": [],
    }


def _write_artifacts(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "sample_index", "device_event_ms", "host_wall_ms"],
        )
        writer.writeheader()
        for row in result["results"]:
            for index, event_ms in enumerate(row.get("samples_ms", [])):
                writer.writerow(
                    {
                        "stage": row["stage"],
                        "sample_index": index,
                        "device_event_ms": event_ms,
                        "host_wall_ms": row["host_samples_ms"][index],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flaggems-root", type=Path, default=ROOT.parent / "FlagGems-stofm")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--stofm-revision")
    parser.add_argument("--flaggems-revision")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ffn-embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--calls-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-resamples", type=int, default=4000)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    if args.nodes <= 0 or args.layers <= 0 or args.heads <= 0:
        raise ValueError("nodes, layers, and heads must be positive")
    if args.gaussian_hidden_dim > 128 or args.heads > 16:
        raise ValueError("the current native Gaussian kernel supports gaussian_hidden_dim <= 128 and heads <= 16")
    library = args.library.resolve()
    if not library.is_file():
        raise FileNotFoundError(f"MUSA extension library does not exist: {library}")
    source_root = (args.flaggems_root.resolve() / "src")
    if not (source_root / "flag_gems").is_dir():
        raise FileNotFoundError(f"FlagGems source package does not exist: {source_root}")
    stofm_revision = _verified_revision(ROOT, args.stofm_revision, "SToFM")
    flaggems_revision = _verified_revision(
        args.flaggems_root, args.flaggems_revision, "FlagGems"
    )
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(ROOT))
    os.environ["FLAGGEMS_STOFM_MUSA_LIBRARY"] = str(library)
    os.environ["FLAGGEMS_STOFM_ENABLE_MUSA_NATIVE"] = "1"
    os.environ["FLAGGEMS_STOFM_REQUIRE_MUSA_NATIVE"] = "1"
    os.environ["FLAGGEMS_STOFM_MUSA_MINIMAL_IMPORT"] = "1"

    import torch_musa  # noqa: F401

    if not torch.musa.is_available():
        raise RuntimeError("torch.musa.is_available() is false")
    from model.se2transformer import SToFMModel
    from flag_gems.experimental_ops.stofm_backends import mthreads

    runtime = torch.musa
    device = torch.device("musa:0")
    dtype = _dtype(args.precision)
    torch.manual_seed(args.seed)
    base_spec = STAGES[0]
    base = SToFMModel(_config(args, base_spec)).to(device=device, dtype=dtype).eval()
    models: Dict[str, Any] = {base_spec.identifier: base}
    for spec in STAGES[1:]:
        candidate = SToFMModel(_config(args, spec)).to(device=device, dtype=dtype).eval()
        candidate.load_state_dict(base.state_dict())
        models[spec.identifier] = candidate
    tokens = torch.randn(args.batch_size, args.nodes, args.input_dim, device=device, dtype=dtype)
    distances = torch.rand(args.batch_size, args.nodes, args.nodes, device=device, dtype=dtype)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(args.batch_size, args.nodes, dtype=torch.long, device=device)
    inputs = (tokens, distances, token_types)

    with torch.inference_mode():
        reference = base(tokens, distances, token_types, return_pair_rep=False)["last_hidden_state"]
        runtime.synchronize()
    reference_digest = _tensor_digest(reference)
    results = []
    for index, spec in enumerate(STAGES):
        model = models[spec.identifier]
        validation = _validate_stage(spec, model, reference, inputs, dtype)
        measured = _measure_inference(
            lambda: model(tokens, distances, token_types, return_pair_rep=False),
            runtime=runtime,
            warmup=args.warmup,
            repetitions=args.repetitions,
            calls_per_sample=args.calls_per_sample,
            seed=args.seed + index * 97,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        results.append(
            {
                "stage": spec.identifier,
                "description": spec.description,
                "baseline_stage": spec.baseline_identifier,
                "status": "measured",
                "expected_custom_operators": list(spec.expected_operators),
                "validation": validation,
                **measured,
            }
        )
    results.append(_unavailable_aten_row())
    properties = runtime.get_device_properties(device)
    native_status = {
        name: mthreads.native_extension_status(name)
        for name in ("gaussian_pair_bias", "pair_score_epilogue")
    }
    result = {
        "schema_version": 1,
        "role": "optimized_musa_native",
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("musa-s4000-%Y%m%dT%H%M%SZ"),
        "trial": args.trial,
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
            "flaggems_package": "flag_gems",
            "musa_library_sha256": _sha256_file(library),
        },
        "workload": {
            "precision": args.precision,
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "layers": args.layers,
            "embedding_dim": args.embedding_dim,
            "ffn_embedding_dim": args.ffn_embedding_dim,
            "heads": args.heads,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "input_dim": args.input_dim,
            "seed": args.seed,
            "input_sha256": {
                "tokens": _tensor_digest(tokens),
                "distances": _tensor_digest(distances),
                "token_types": _tensor_digest(token_types),
                "reference_last_hidden_state": reference_digest,
            },
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
        "native_extension": native_status,
        "results": results,
    }
    _write_artifacts(args.output_dir, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
