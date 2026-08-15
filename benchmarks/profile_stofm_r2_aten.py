"""Capture R2 SToFM ATen coverage without mixing profiling into timing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.se2transformer import SToFMModel
from model.utils import SToFMConfig
from r2_benchmark_common import git_sha, jsonable, runtime_capture


ATEN_CLASSIFICATION = {
    "aten::addmm": "stock_flagos_aten",
    "aten::baddbmm": "stock_flagos_aten",
    "aten::bmm": "stock_flagos_aten",
    "aten::_softmax": "stock_flagos_aten",
    "aten::softmax": "stock_flagos_aten",
    "aten::layer_norm": "torch_retained_candidate_rejected_pending_profile",
    "aten::native_layer_norm": "torch_retained_candidate_rejected_pending_profile",
    "aten::gelu": "torch_retained",
    "aten::silu": "vision_swiglu_candidate",
}


def _stage_config(stage: str, args: argparse.Namespace) -> SToFMConfig:
    mode, gaussian, attention = {
        "p1": ("torch", "torch", "torch"),
        "f0": ("stock", "torch", "torch"),
        "final": ("optimized", "flaggems", "flaggems"),
    }[stage]
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
        flagos_mode=mode,
        flagos_backend=gaussian,
        flagos_attention_backend=attention,
    )


def _event_value(event: Any, attribute: str) -> float:
    value = getattr(event, attribute, 0.0)
    return float(value) if value is not None else 0.0


def _profile_rows(profile, limit: int) -> List[Dict[str, Any]]:
    rows = []
    for event in profile.key_averages():
        cuda_us = _event_value(event, "self_device_time_total")
        if not cuda_us:
            cuda_us = _event_value(event, "self_cuda_time_total")
        cpu_us = _event_value(event, "self_cpu_time_total")
        rows.append(
            {
                "name": event.key,
                "count": int(getattr(event, "count", 0)),
                "self_cuda_us": cuda_us,
                "self_cpu_us": cpu_us,
                "classification": ATEN_CLASSIFICATION.get(event.key, "unclassified"),
            }
        )
    rows.sort(key=lambda row: (row["self_cuda_us"], row["self_cpu_us"]), reverse=True)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["p1", "f0", "final"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flaggems-source-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1050)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ffn-embedding-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--gaussian-hidden-dim", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SToFM R2 profiling requires CUDA")
    device = torch.device(f"cuda:{args.device_index}")
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = SToFMModel(_stage_config(args.stage, args)).to(device).eval()
    if dtype == torch.float16:
        model.half()
    tokens = torch.randn(args.batch_size, args.nodes, args.input_dim, device=device, dtype=dtype)
    distances = torch.rand(args.batch_size, args.nodes, args.nodes, device=device, dtype=dtype)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(args.batch_size, args.nodes, dtype=torch.long, device=device)

    def invoke():
        return model(tokens, distances, token_types, return_pair_rep=False)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.inference_mode():
        if args.stage == "p1":
            invoke()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
                invoke()
            dispatch = {"runtime": model.last_flagos_runtime_dispatch}
        else:
            with model.flagos_inference_scope() as outer_scope:
                invoke()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profile:
                    invoke()
                dispatch = {
                    "outer_scope": outer_scope,
                    "runtime": model.last_flagos_runtime_dispatch,
                    "gaussian": model.gaussian.last_flagos_dispatch,
                    "pair_attention_layer0": model.encoder.layers[0].self_attn.last_flagos_dispatch,
                }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(args.output_dir / "trace.json"))
    result = {
        "schema_version": 1,
        "stage": args.stage,
        "precision": args.precision,
        "runtime": runtime_capture(device),
        "commits": {"stofm": git_sha(ROOT), "flaggems": git_sha(args.flaggems_source_root)},
        "workload": {
            "batch_size": args.batch_size,
            "nodes": args.nodes,
            "layers": args.layers,
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "gaussian_hidden_dim": args.gaussian_hidden_dim,
            "input_dim": args.input_dim,
            "return_pair_rep": False,
        },
        "dispatch": jsonable(dispatch),
        "events": _profile_rows(profile, args.top),
        "trace": "trace.json",
    }
    (args.output_dir / "profile.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SToFM R2 ATen Profile",
        "",
        f"Stage: `{args.stage}`; precision: `{args.precision}`.",
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
