"""Reproducible V100 microbenchmarks for experimental Uni2/KRONOS operators."""

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
import statistics
import subprocess
from typing import Callable, Dict, Iterable, List

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


def _benchmark(name: str, baseline_stage: str, fn: Callable[[], object], warmup: int, repetitions: int, calls: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    samples: List[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return {
        "stage": name,
        "baseline_stage": baseline_stage,
        "status": "measured",
        "samples_ms": samples,
        "p20_ms": _quantile(samples, 0.20),
        "p50_ms": _quantile(samples, 0.50),
        "p80_ms": _quantile(samples, 0.80),
        "p95_ms": _quantile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
        "peak_delta_allocated_mib": (torch.cuda.max_memory_allocated() - allocated) / 1024**2,
    }


def _write_reports(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result["results"]
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "sample_index", "latency_ms"])
        writer.writeheader()
        for row in rows:
            for index, latency in enumerate(row["samples_ms"]):
                writer.writerow({"stage": row["stage"], "sample_index": index, "latency_ms": latency})

    measured = {row["stage"]: row for row in rows if row["status"] == "measured"}
    lines = [
        "# Vision Operator V100 Benchmark",
        "",
        f"Run ID: `{result['run_id']}`",
        "",
        "| Stage | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Relative p50 | Peak delta MiB | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    html_rows = []
    for row in rows:
        if row["status"] == "measured":
            baseline = measured[row["baseline_stage"]]
            relative = f"{baseline['p50_ms'] / row['p50_ms']:.3f}x"
            sample_count = str(len(row["samples_ms"]))
            values = [
                sample_count,
                f"{row['p20_ms']:.4f}",
                f"{row['p50_ms']:.4f}",
                f"{row['p80_ms']:.4f}",
                f"{row['p95_ms']:.4f}",
                f"{row['mean_ms']:.4f}",
                relative,
                f"{row['peak_delta_allocated_mib']:.1f}",
                row["status"],
            ]
        else:
            values = ["-", "-", "-", "-", "-", "-", "-", "-", row["status"]]
        lines.append(f"| {row['stage']} | " + " | ".join(values) + " |")
        html_rows.append("<tr><td>" + html.escape(row["stage"]) + "</td>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = "".join(html_rows)
    (output_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Vision Operator V100 Benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem .7rem;border:1px solid #bbb}</style>"
        "<h1>Vision Operator V100 Benchmark</h1><table><tr><th>Stage</th><th>Samples</th><th>p20 ms</th><th>p50 ms</th><th>p80 ms</th><th>p95 ms</th><th>Mean ms</th><th>Relative p50</th><th>Peak delta MiB</th><th>Status</th></tr>"
        + table
        + "</table>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results" / "vision-v100")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The V100 vision benchmark requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    patches = torch.randn(
        args.batch_size,
        args.markers,
        args.tokens_per_marker,
        args.embedding_dim,
        device=device,
    )
    marker_ids = torch.arange(args.markers, device=device, dtype=torch.long).remainder(args.marker_vocab)
    marker_ids = marker_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    marker_padding = torch.zeros(args.batch_size, args.markers, dtype=torch.bool, device=device)
    marker_padding[:, -(args.markers // 4) :] = True
    marker_ids = marker_ids.masked_fill(marker_padding, -1)
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
            marker_padding_mask=marker_padding,
            backend="torch",
        )[0]

    def marker_native():
        return marker_token_embed(
            patches,
            marker_ids,
            marker_weight,
            position_embedding=position,
            token_embedding=token,
            marker_padding_mask=marker_padding,
            backend="nvidia",
        )[0]

    with torch.inference_mode():
        torch.testing.assert_close(marker_native(), marker_reference(), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(vit_swiglu(packed_swiglu, backend="nvidia"), vit_swiglu(packed_swiglu, backend="torch"), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(
            vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend="nvidia"),
            vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend="torch"),
            rtol=3e-4,
            atol=3e-5,
        )
        results = [
            _benchmark("B0_marker_token_reference", "B0_marker_token_reference", marker_reference, args.warmup, args.repetitions, args.calls_per_sample),
            _benchmark("O1_marker_token_triton", "B0_marker_token_reference", marker_native, args.warmup, args.repetitions, args.calls_per_sample),
            _benchmark("B0_swiglu_reference", "B0_swiglu_reference", lambda: vit_swiglu(packed_swiglu, backend="torch"), args.warmup, args.repetitions, args.calls_per_sample),
            _benchmark("O2_swiglu_existing_triton", "B0_swiglu_reference", lambda: vit_swiglu(packed_swiglu, backend="nvidia"), args.warmup, args.repetitions, args.calls_per_sample),
            _benchmark("B0_residual_layer_norm", "B0_residual_layer_norm", lambda: vit_residual_layer_norm(residual_input, residual, (args.embedding_dim,), norm_weight, norm_bias, backend="torch"), args.warmup, args.repetitions, args.calls_per_sample),
            {
                "stage": "O3_residual_layer_norm",
                "baseline_stage": "B0_residual_layer_norm",
                "status": "rejected",
                "reason": "The existing skip_layer_norm candidate is slower on V100 and has no verified backward path.",
                "samples_ms": [],
            },
        ]
    result = {
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("vision-v100-%Y%m%dT%H%M%SZ"),
        "hardware": {
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "workload": {**vars(args), "output_dir": str(args.output_dir)},
        "commits": {
            "stofm": _git_sha(ROOT),
            "flaggems": _git_sha(ROOT.parent / "FlagGems-stofm"),
        },
        "validation": {"status": "passed", "rtol": 3e-4, "atol": 3e-5, "mode": "torch.inference_mode"},
        "results": results,
    }
    _write_reports(args.output_dir, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
