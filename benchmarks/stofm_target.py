"""Correctness-first SToFM benchmark harness for rented target accelerators.

This module intentionally has no import-time torch_npu or torch_musa dependency.
Run it only after the corresponding vendor PyTorch extension has registered the
``npu`` or ``musa`` device type. It emits the same result.json/samples.csv
contract as the V100 harness so independent target runs can be aggregated.
"""

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.se2transformer import SToFMModel
from model.utils import SToFMConfig


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
            f"torch.{device_type} is unavailable. Install the target vendor PyTorch extension before running this harness."
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


def _config(args, backend: str, attention_backend: Optional[str] = None) -> SToFMConfig:
    return SToFMConfig(
        num_hidden_layers=args.layers,
        embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.embedding_dim,
        num_attention_heads=args.heads,
        gaussian_hidden_dim=args.gaussian_hidden_dim,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        input_dim=args.input_dim,
        flagos_backend=backend,
        flagos_attention_backend=attention_backend,
    )


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
        "scope": "end_to_end",
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
        "# SToFM Target Benchmark",
        "",
        f"Run ID: `{result['run_id']}`",
        "",
        "| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Peak allocated MiB | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
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
            row["status"],
        ]
        lines.append(f"| {row['stage']} | " + " | ".join(values) + " |")
        html_rows.append("<tr><td>" + html.escape(row["stage"]) + "</td>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>SToFM Target Benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem .7rem;border:1px solid #bbb}</style>"
        "<h1>SToFM Target Benchmark</h1><table><tr><th>Stage</th><th>Samples</th><th>p20 ms</th><th>p50 ms</th><th>p80 ms</th><th>p95 ms</th><th>Mean ms</th><th>Peak allocated MiB</th><th>Status</th></tr>"
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
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_backend = "ascend" if args.device == "npu" else "mthreads"
    if args.backend != expected_backend:
        raise ValueError(f"--device {args.device} requires --backend {expected_backend}")
    runtime = _target_runtime(args.device)
    device = torch.device(f"{args.device}:{args.device_index}")
    torch.manual_seed(args.seed)

    base_config = _config(args, "torch")
    target_gaussian_config = _config(args, args.backend, attention_backend="torch")
    target_attention_config = _config(args, "torch", attention_backend=args.backend)
    target_combined_config = _config(args, args.backend)
    base = SToFMModel(base_config).to(device).eval()
    target_gaussian = SToFMModel(target_gaussian_config).to(device).eval()
    target_attention = SToFMModel(target_attention_config).to(device).eval()
    target_combined = SToFMModel(target_combined_config).to(device).eval()
    for model in (target_gaussian, target_attention, target_combined):
        model.load_state_dict(base.state_dict())

    embeddings = torch.randn(args.batch_size, args.nodes, args.input_dim, device=device)
    distances = torch.rand(args.batch_size, args.nodes, args.nodes, device=device)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(args.batch_size, args.nodes, dtype=torch.long, device=device)
    with torch.inference_mode():
        reference = base(embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"]
        outputs = {
            "O1_target_gaussian": target_gaussian(embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"],
            "O2_target_attention": target_attention(embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"],
            "O5_target_combined": target_combined(embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"],
        }
        for name, output in outputs.items():
            try:
                torch.testing.assert_close(output, reference, rtol=3e-4, atol=3e-5)
            except AssertionError as exc:
                raise RuntimeError(f"Target correctness gate failed: {name}") from exc
        results = [
            _benchmark("B0_e2e", "B0_e2e", lambda: base(embeddings, distances, token_types, return_pair_rep=True), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("B1_e2e", "B0_e2e", lambda: base(embeddings, distances, token_types, return_pair_rep=False), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O1_e2e_target_gaussian", "B1_e2e", lambda: target_gaussian(embeddings, distances, token_types, return_pair_rep=False), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O2_e2e_target_attention", "B1_e2e", lambda: target_attention(embeddings, distances, token_types, return_pair_rep=False), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
            _benchmark("O5_e2e_target_combined", "B1_e2e", lambda: target_combined(embeddings, distances, token_types, return_pair_rep=False), runtime=runtime, device=device, warmup=args.warmup, repetitions=args.repetitions, calls_per_sample=args.calls_per_sample),
        ]

    get_device_name = getattr(runtime, "get_device_name", None)
    try:
        device_name = str(get_device_name(args.device_index)) if get_device_name is not None else args.device
    except (RuntimeError, TypeError):
        device_name = args.device
    result = {
        "run_id": dt.datetime.now(dt.timezone.utc).strftime(f"{args.backend}-%Y%m%dT%H%M%SZ"),
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
