"""Shared evidence utilities for the R2 CUDA benchmark workers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List

import torch


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def quantile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize_samples(samples: List[float]) -> Dict[str, float]:
    return {
        "p20_ms": quantile(samples, 0.20),
        "p50_ms": quantile(samples, 0.50),
        "p80_ms": quantile(samples, 0.80),
        "p95_ms": quantile(samples, 0.95),
        "mean_ms": statistics.mean(samples),
    }


def benchmark_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
    calls_per_sample: int,
) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    samples: List[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls_per_sample)
    result: Dict[str, Any] = {
        "samples_ms": samples,
        "sample_count": len(samples),
        "calls_per_sample": calls_per_sample,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_delta_allocated_mib": (
            torch.cuda.max_memory_allocated() - allocated_before
        )
        / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "peak_delta_reserved_mib": (
            torch.cuda.max_memory_reserved() - reserved_before
        )
        / 1024**2,
    }
    result.update(summarize_samples(samples))
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def pip_freeze() -> List[str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line for line in output.splitlines() if line)


def nvidia_smi() -> List[Dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in output.splitlines():
        index, name, driver, memory_mib = (part.strip() for part in line.split(",", maxsplit=3))
        rows.append(
            {
                "index": index,
                "name": name,
                "driver": driver,
                "memory_total_mib": memory_mib,
            }
        )
    return rows


def runtime_capture(device: torch.device) -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "nvidia_smi": nvidia_smi(),
        "pip_freeze": pip_freeze(),
    }
