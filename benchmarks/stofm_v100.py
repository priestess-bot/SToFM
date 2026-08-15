"""Reproducible V100 benchmark for the SToFM FlagGems integration."""

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, Optional

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.se2transformer import GaussianModule, MultiheadAttention, SToFMModel
from model.utils import SToFMConfig
from flag_gems.experimental_ops import stofm_gaussian_pair_bias


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _nvidia_smi() -> List[Dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    devices = []
    for line in output.splitlines():
        name, driver, memory_mib = (part.strip() for part in line.split(",", maxsplit=2))
        devices.append({"name": name, "driver": driver, "memory_total_mib": memory_mib})
    return devices


def _quantile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _benchmark(
    name: str,
    scope: str,
    baseline_stage: str,
    fn: Callable[[], object],
    warmup: int,
    repetitions: int,
    calls_per_sample: int,
) -> Dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls_per_sample)
    return {
        "stage": name,
        "scope": scope,
        "baseline_stage": baseline_stage,
        "status": "measured",
        "samples_ms": samples,
        "p20_ms": _quantile(samples, 0.20),
        "p50_ms": _quantile(samples, 0.50),
        "p80_ms": _quantile(samples, 0.80),
        "p95_ms": _quantile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_delta_allocated_mib": (torch.cuda.max_memory_allocated() - baseline_allocated) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "peak_delta_reserved_mib": (torch.cuda.max_memory_reserved() - baseline_reserved) / 1024**2,
    }


def _config(
    backend: str,
    layers: int,
    embedding_dim: int,
    heads: int,
    gaussian_hidden_dim: int,
    input_dim: int,
    attention_backend: Optional[str] = None,
):
    return SToFMConfig(
        num_hidden_layers=layers,
        embedding_dim=embedding_dim,
        ffn_embedding_dim=embedding_dim,
        num_attention_heads=heads,
        gaussian_hidden_dim=gaussian_hidden_dim,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        input_dim=input_dim,
        flagos_backend=backend,
        flagos_attention_backend=attention_backend,
    )


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

    markdown_lines = [
        "# SToFM V100 Benchmark",
        "",
        f"Run ID: `{result['run_id']}`",
        "",
        f"Device: `{result['hardware']['name']}`; PyTorch `{result['hardware']['torch']}`; CUDA `{result['hardware']['cuda']}`",
        "",
        "Correctness gate: `passed` for B1/O3/O4/O5 end-to-end last hidden state before timing.",
        "",
        "| Stage | Scope | Samples | p20 ms | p50 ms | p80 ms | p95 ms | Mean ms | Speedup | Peak delta MiB | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    html_rows = []
    measured = {row["stage"]: row for row in rows if row["status"] == "measured"}
    for row in rows:
        if row["status"] == "measured":
            baseline = measured[row["baseline_stage"]]
            speedup = baseline["p50_ms"] / row["p50_ms"]
            samples = str(len(row["samples_ms"]))
            p20 = f"{row['p20_ms']:.4f}"
            p50 = f"{row['p50_ms']:.4f}"
            p80 = f"{row['p80_ms']:.4f}"
            p95 = f"{row['p95_ms']:.4f}"
            mean = f"{row['mean_ms']:.4f}"
            peak = f"{row['peak_delta_allocated_mib']:.1f}"
            speedup_text = f"{speedup:.3f}x"
        else:
            samples = "-"
            p20 = "-"
            p50 = "-"
            p80 = "-"
            p95 = "-"
            mean = "-"
            peak = "-"
            speedup_text = "-"
        markdown_lines.append(
            f"| {row['stage']} | {row['scope']} | {samples} | {p20} | {p50} | {p80} | {p95} | {mean} | {speedup_text} | {peak} | {row['status']} |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(row['stage'])}</td><td>{html.escape(row['scope'])}</td><td>{samples}</td>"
            f"<td>{p20}</td><td>{p50}</td><td>{p80}</td><td>{p95}</td><td>{mean}</td>"
            f"<td>{speedup_text}</td><td>{peak}</td><td>{html.escape(row['status'])}</td></tr>"
        )
    (output_dir / "report.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    report_html = (
        "<!doctype html><meta charset='utf-8'><title>SToFM V100 Benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.4rem .7rem;border:1px solid #bbb}</style>"
        "<h1>SToFM V100 Benchmark</h1><table><tr><th>Stage</th><th>Scope</th><th>Samples</th><th>p20 ms</th><th>p50 ms</th><th>p80 ms</th><th>p95 ms</th><th>Mean ms</th><th>Speedup</th><th>Peak delta MiB</th><th>Status</th></tr>"
        + "".join(html_rows)
        + "</table>"
    )
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results" / "v100")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The V100 benchmark requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    base_config = _config("torch", args.layers, args.embedding_dim, args.heads, args.gaussian_hidden_dim, args.input_dim)
    fast_config = _config("flaggems", args.layers, args.embedding_dim, args.heads, args.gaussian_hidden_dim, args.input_dim)
    fast_native_attention_config = _config(
        "flaggems",
        args.layers,
        args.embedding_dim,
        args.heads,
        args.gaussian_hidden_dim,
        args.input_dim,
        attention_backend="torch",
    )
    attention_reference_config = _config(
        "torch",
        args.layers,
        args.embedding_dim,
        args.heads,
        args.gaussian_hidden_dim,
        args.input_dim,
        attention_backend="inductor",
    )
    attention_native_config = _config(
        "torch",
        args.layers,
        args.embedding_dim,
        args.heads,
        args.gaussian_hidden_dim,
        args.input_dim,
        attention_backend="nvidia",
    )
    fast_reference_pair_config = _config(
        "flaggems",
        args.layers,
        args.embedding_dim,
        args.heads,
        args.gaussian_hidden_dim,
        args.input_dim,
        attention_backend="inductor",
    )
    fast_native_epilogue_config = _config(
        "flaggems",
        args.layers,
        args.embedding_dim,
        args.heads,
        args.gaussian_hidden_dim,
        args.input_dim,
        attention_backend="nvidia",
    )
    gaussian_base = GaussianModule(base_config).to(device).eval()
    gaussian_fast = GaussianModule(fast_config).to(device).eval()
    gaussian_fast.load_state_dict(gaussian_base.state_dict())
    attention_base = MultiheadAttention(base_config).to(device).eval()
    attention_reference = MultiheadAttention(attention_reference_config).to(device).eval()
    attention_native = MultiheadAttention(attention_native_config).to(device).eval()
    attention_reference.load_state_dict(attention_base.state_dict())
    attention_native.load_state_dict(attention_base.state_dict())
    model_base = SToFMModel(base_config).to(device).eval()
    model_fast = SToFMModel(fast_config).to(device).eval()
    model_fast_reference_pair = SToFMModel(fast_reference_pair_config).to(device).eval()
    model_fast_native_attention = SToFMModel(fast_native_attention_config).to(device).eval()
    model_fast_native_epilogue = SToFMModel(fast_native_epilogue_config).to(device).eval()
    model_fast.load_state_dict(model_base.state_dict())
    model_fast_reference_pair.load_state_dict(model_base.state_dict())
    model_fast_native_attention.load_state_dict(model_base.state_dict())
    model_fast_native_epilogue.load_state_dict(model_base.state_dict())

    distances = torch.rand(args.batch_size, args.nodes, args.nodes, device=device)
    distances[:, 0, 0] = 0.0
    query = torch.randn(args.nodes, args.batch_size, args.embedding_dim, device=device)
    pair_bias = gaussian_base(distances)
    token_embeddings = torch.randn(args.batch_size, args.nodes, args.input_dim, device=device)
    token_types = torch.zeros(args.batch_size, args.nodes, dtype=torch.long, device=device)

    def native_gaussian():
        return stofm_gaussian_pair_bias(
            distances,
            gaussian_base.linear.weight,
            gaussian_base.linear.bias,
            gaussian_base.means.weight,
            gaussian_base.stds.weight,
            gaussian_base.proj[0].weight,
            gaussian_base.proj[0].bias,
            gaussian_base.proj[2].weight,
            gaussian_base.proj[2].bias,
            zero_mask=distances.eq(0.0),
            backend="nvidia",
        )

    with torch.inference_mode():
        native_gaussian_output = native_gaussian()
        torch.testing.assert_close(
            native_gaussian_output,
            gaussian_base._forward_torch(distances),
            rtol=3e-4,
            atol=3e-5,
        )
        reference_e2e = model_base(token_embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"]
        native_attention_e2e = model_fast_native_attention(
            token_embeddings, distances, token_types, return_pair_rep=False
        )["last_hidden_state"]
        pair_attention_e2e = model_fast_reference_pair(
            token_embeddings, distances, token_types, return_pair_rep=False
        )["last_hidden_state"]
        default_e2e = model_fast(token_embeddings, distances, token_types, return_pair_rep=False)["last_hidden_state"]
        native_epilogue_e2e = model_fast_native_epilogue(
            token_embeddings, distances, token_types, return_pair_rep=False
        )["last_hidden_state"]
        torch.testing.assert_close(native_attention_e2e, reference_e2e, rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(pair_attention_e2e, reference_e2e, rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(default_e2e, reference_e2e, rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(native_epilogue_e2e, reference_e2e, rtol=3e-4, atol=3e-5)

        results = [
            _benchmark(
                "B0_gaussian", "gaussian", "B0_gaussian", lambda: gaussian_base._forward_torch(distances),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O1_gaussian", "gaussian", "B0_gaussian", lambda: gaussian_fast(distances),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O1n_gaussian_triton", "gaussian", "O1_gaussian", native_gaussian,
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "B0_attention", "attention", "B0_attention",
                lambda: attention_base(query, query, query, pair_bias, need_weights=False, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O2_attention", "attention", "B0_attention",
                lambda: attention_reference(query, query, query, pair_bias, need_weights=False, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O2n_attention_triton_epilogue", "attention", "O2_attention",
                lambda: attention_native(query, query, query, pair_bias, need_weights=False, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "B0_e2e", "end_to_end", "B0_e2e",
                lambda: model_base(token_embeddings, distances, token_types, return_pair_rep=True),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "B1_e2e", "end_to_end", "B0_e2e",
                lambda: model_base(token_embeddings, distances, token_types, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            {
                "stage": "B2_e2e",
                "scope": "end_to_end",
                "baseline_stage": "B1_e2e",
                "status": "skipped",
                "reason": "Generic use_gems patching is intentionally excluded from the semantic direct-op benchmark.",
                "samples_ms": [],
            },
            _benchmark(
                "O3_e2e_native_attention", "end_to_end", "B1_e2e",
                lambda: model_fast_native_attention(token_embeddings, distances, token_types, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O4_e2e_pair_attention", "end_to_end", "B1_e2e",
                lambda: model_fast_reference_pair(token_embeddings, distances, token_types, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
            _benchmark(
                "O5_e2e_triton_pair_epilogue", "end_to_end", "O4_e2e_pair_attention",
                lambda: model_fast(token_embeddings, distances, token_types, return_pair_rep=False),
                args.warmup, args.repetitions, args.calls_per_sample,
            ),
        ]

    result = {
        "run_id": dt.datetime.now(dt.timezone.utc).strftime("v100-%Y%m%dT%H%M%SZ"),
        "hardware": {
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi": _nvidia_smi(),
        },
        "validation": {
            "native_gaussian": {
                "status": "passed",
                "rtol": 3e-4,
                "atol": 3e-5,
                "reference": "B0_gaussian",
                "candidate": "O1n_gaussian_triton",
            },
            "end_to_end_last_hidden_state": {
                "status": "passed",
                "rtol": 3e-4,
                "atol": 3e-5,
                "reference": "B1_e2e",
                "candidates": [
                    "O3_e2e_native_attention",
                    "O4_e2e_pair_attention",
                    "O5_e2e_triton_pair_epilogue",
                ],
            },
            "measurement": {
                "timer": "CUDA events",
                "compile_included": False,
                "mode": "torch.inference_mode",
                "dropout": 0.0,
            },
        },
        "workload": {**vars(args), "output_dir": str(args.output_dir)},
        "commits": {
            "stofm": _git_sha(ROOT),
            "flaggems": _git_sha(ROOT.parent / "FlagGems-stofm"),
        },
        "results": results,
    }
    _write_reports(args.output_dir, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
