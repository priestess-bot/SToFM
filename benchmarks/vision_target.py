"""Correctness-first Uni2/KRONOS operator benchmark for rented accelerators.

The target vendor extension must register ``npu`` or ``musa`` before this
script is run. There are no import-time vendor dependencies, so the source can
be syntax-checked on the V100 development host.
"""

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from flag_gems.experimental_ops import marker_token_embed, vit_residual_layer_norm, vit_swiglu


ROOT = Path(__file__).parents[1]


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _quantile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _target_runtime(device_type: str):
    runtime = getattr(torch, device_type, None)
    if runtime is None or not getattr(runtime, "is_available", lambda: False)():
        raise RuntimeError(
            f"torch.{device_type} is unavailable. Install the target vendor PyTorch extension before running this benchmark."
        )
    return runtime


def _synchronize(runtime: Any, device: torch.device) -> None:
    try:
        runtime.synchronize(device)
    except TypeError:
        runtime.synchronize()


def _memory_value(runtime: Any, name: str) -> Optional[float]:
    function = getattr(runtime, name, None)
    if function is None:
        return None
    try:
        return float(function()) / 1024**2
    except (RuntimeError, TypeError):
        return None


def _reset_peak_memory(runtime: Any) -> None:
    function = getattr(runtime, "reset_peak_memory_stats", None)
    if function is not None:
        try:
            function()
        except (RuntimeError, TypeError):
            pass


def _benchmark(
    stage: str,
    baseline_stage: str,
    fn: Callable[[], object],
    *,
    runtime: Any,
    device: torch.device,
    warmup: int,
    repetitions: int,
    calls_per_sample: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    _synchronize(runtime, device)
    _reset_peak_memory(runtime)
    baseline_allocated = _memory_value(runtime, "memory_allocated")
    samples = []
    for _ in range(repetitions):
        _synchronize(runtime, device)
        start = time.perf_counter()
        for _ in range(calls_per_sample):
            fn()
        _synchronize(runtime, device)
        samples.append((time.perf_counter() - start) * 1000.0 / calls_per_sample)
    peak_allocated = _memory_value(runtime, "max_memory_allocated")
    return {
        "stage": stage,
        "scope": "vision_operator",
        "baseline_stage": baseline_stage,
        "status": "measured",
        "samples_ms": samples,
        "p20_ms": _quantile(samples, 0.20),
        "p50_ms": _quantile(samples, 0.50),
        "p80_ms": _quantile(samples, 0.80),
        "p95_ms": _quantile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
        "peak_allocated_mib": peak_allocated,
        "peak_delta_allocated_mib": (
            peak_allocated - baseline_allocated if peak_allocated is not None and baseline_allocated is not None else None
        ),
    }


def _write_reports(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "sample_index", "latency_ms"])
        writer.writeheader()
        for row in result["results"]:
            for index, latency in enumerate(row["samples_ms"]):
                writer.writerow({"stage": row["stage"], "sample_index": index, "latency_ms": latency})
    lines = [
        "# Vision Target Benchmark",
        "",
        f"Run ID: `{result['run_id']}`",
        "",
        "| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Peak allocated MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        peak = "unavailable" if row["peak_allocated_mib"] is None else f"{row['peak_allocated_mib']:.1f}"
        lines.append(
            f"| {row['stage']} | {len(row['samples_ms'])} | {row['p20_ms']:.4f} | {row['p50_ms']:.4f} | "
            f"{row['p80_ms']:.4f} | {row['p95_ms']:.4f} | {row['mean_ms']:.4f} | {peak} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_rows = []
    for row in result["results"]:
        peak = "unavailable" if row["peak_allocated_mib"] is None else f"{row['peak_allocated_mib']:.1f}"
        values = [
            str(len(row["samples_ms"])),
            f"{row['p20_ms']:.4f}",
            f"{row['p50_ms']:.4f}",
            f"{row['p80_ms']:.4f}",
            f"{row['p95_ms']:.4f}",
            f"{row['mean_ms']:.4f}",
            peak,
        ]
        html_rows.append("<tr><td>" + html.escape(row["stage"]) + "</td>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    (output_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Vision Target Benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem .7rem;border:1px solid #bbb}</style>"
        "<h1>Vision Target Benchmark</h1><table><tr><th>Stage</th><th>Samples</th><th>p20 ms</th><th>p50 ms</th><th>p80 ms</th><th>p95 ms</th><th>Mean ms</th><th>Peak allocated MiB</th></tr>"
        + "".join(html_rows)
        + "</table>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["npu", "musa"], required=True)
    parser.add_argument("--backend", choices=["ascend", "mthreads"], required=True)
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
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_backend = "ascend" if args.device == "npu" else "mthreads"
    if args.backend != expected_backend:
        raise ValueError(f"--device {args.device} requires --backend {expected_backend}")
    runtime = _target_runtime(args.device)
    torch.manual_seed(args.seed)
    device = torch.device(f"{args.device}:{args.device_index}")

    patches = torch.randn(args.batch_size, args.markers, args.tokens_per_marker, args.embedding_dim, device=device)
    marker_ids = torch.arange(args.markers, device=device, dtype=torch.long).remainder(args.marker_vocab)
    marker_ids = marker_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    padding = torch.zeros(args.batch_size, args.markers, dtype=torch.bool, device=device)
    padding[:, -(args.markers // 4) :] = True
    marker_ids = marker_ids.masked_fill(padding, -1)
    marker_weight = torch.randn(args.marker_vocab, args.embedding_dim, device=device)
    position = torch.randn(args.tokens_per_marker, args.embedding_dim, device=device)
    token = torch.randn(args.tokens_per_marker, args.embedding_dim, device=device)
    packed_swiglu = torch.randn(args.batch_size, args.swiglu_sequence, 2 * args.swiglu_hidden, device=device)
    residual_input = torch.randn(args.batch_size, args.swiglu_sequence, args.embedding_dim, device=device)
    residual = torch.randn_like(residual_input)
    norm_weight = torch.randn(args.embedding_dim, device=device)
    norm_bias = torch.randn(args.embedding_dim, device=device)

    def marker_reference():
        return marker_token_embed(
            patches,
            marker_ids,
            marker_weight,
            position_embedding=position,
            token_embedding=token,
            marker_padding_mask=padding,
            backend="torch",
        )[0]

    def marker_target():
        return marker_token_embed(
            patches,
            marker_ids,
            marker_weight,
            position_embedding=position,
            token_embedding=token,
            marker_padding_mask=padding,
            backend=args.backend,
        )[0]

    with torch.inference_mode():
        torch.testing.assert_close(marker_target(), marker_reference(), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(vit_swiglu(packed_swiglu, backend=args.backend), vit_swiglu(packed_swiglu, backend="torch"), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(
            vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend=args.backend),
            vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend="torch"),
            rtol=3e-4,
            atol=3e-5,
        )
        results = [
            _benchmark("B0_marker_token_reference", "B0_marker_token_reference", marker_reference, runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O1_marker_token_target", "B0_marker_token_reference", marker_target, runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("B0_swiglu_reference", "B0_swiglu_reference", lambda: vit_swiglu(packed_swiglu, backend="torch"), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O2_swiglu_target", "B0_swiglu_reference", lambda: vit_swiglu(packed_swiglu, backend=args.backend), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("B0_residual_layer_norm", "B0_residual_layer_norm", lambda: vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend="torch"), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O3_residual_layer_norm_target", "B0_residual_layer_norm", lambda: vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend=args.backend), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
        ]

    get_device_name = getattr(runtime, "get_device_name", None)
    try:
        device_name = str(get_device_name(args.device_index)) if get_device_name is not None else args.device
    except (RuntimeError, TypeError):
        device_name = args.device
    result = {
        "run_id": dt.datetime.now(dt.timezone.utc).strftime(f"vision-{args.backend}-%Y%m%dT%H%M%SZ"),
        "hardware": {"name": device_name, "device_type": args.device, "torch": torch.__version__, "cuda": torch.version.cuda},
        "workload": {**vars(args), "output_dir": str(args.output_dir)},
        "commits": {"stofm": _git_sha(ROOT), "flaggems": _git_sha(ROOT.parent / "FlagGems-stofm")},
        "validation": {"status": "passed", "rtol": 3e-4, "atol": 3e-5, "mode": "torch.inference_mode"},
        "measurement": {"timer": "host_perf_counter_with_device_synchronization", "compile_included": False},
        "results": results,
    }
    _write_reports(args.output_dir, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
