"""Static checks for the two-environment R2 benchmark contract."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read_requirements(name: str) -> str:
    return (ROOT / "requirements" / name).read_text(encoding="utf-8")


def test_r2_stock_and_optimized_manifests_share_torch_cuda_stack():
    base = _read_requirements("flagos-r2-v100.txt")
    assert "torch==2.6.0+cu124" in base
    assert "triton==3.2.0" in base
    assert "https://download.pytorch.org/whl/cu124" in base

    stock = _read_requirements("flagos-r2-stock.txt")
    optimized = _read_requirements("flagos-r2-optimized.txt")
    assert "-r flagos-r2-v100.txt" in stock
    assert "-r flagos-r2-v100.txt" in optimized
    assert "03bf364ede763d573d5c30124d554283a209ab85" in stock
    assert "399d0381ed63a79018f3112ecc43894fd58ba052" in optimized


def test_r2_locks_match_the_frozen_and_optimized_install_specs():
    stock = json.loads((ROOT / "deps" / "flagos-stock.lock.json").read_text(encoding="utf-8"))
    optimized = json.loads((ROOT / "deps" / "flagos-optimized.lock.json").read_text(encoding="utf-8"))
    environment = json.loads((ROOT / "deps" / "v100-r2-environment.json").read_text(encoding="utf-8"))

    assert stock["role"] == "frozen-stock-baseline"
    assert stock["commit"] == "03bf364ede763d573d5c30124d554283a209ab85"
    assert stock["required_environment"] == optimized["required_environment"]
    assert optimized["commit"] == "399d0381ed63a79018f3112ecc43894fd58ba052"
    assert environment["runtime"]["torch"] == optimized["required_environment"]["torch"]
    assert environment["runtime"]["cuda"] == optimized["required_environment"]["cuda"]
