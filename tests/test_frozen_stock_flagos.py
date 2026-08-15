"""F0 contract test runnable with the frozen pre-SToFM FlagGems package."""

import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.flagos_runtime import current_flagos_runtime_dispatch
from model.se2transformer import SToFMModel
from model.utils import SToFMConfig


def _config(mode: str) -> SToFMConfig:
    return SToFMConfig(
        num_hidden_layers=1,
        embedding_dim=32,
        ffn_embedding_dim=64,
        num_attention_heads=4,
        gaussian_hidden_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        input_dim=16,
        flagos_mode=mode,
        # F0 must ignore this candidate selection: the frozen package has no
        # SToFM experimental API and only installs its ATen implementations.
        flagos_backend="nvidia",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="F0 frozen FlagGems validation requires CUDA")
def test_frozen_stock_flaggems_matches_canonical_torch_inference():
    torch.manual_seed(107)
    device = torch.device("cuda")
    baseline = SToFMModel(_config("torch")).to(device).eval()
    stock = SToFMModel(_config("stock")).to(device).eval()
    stock.load_state_dict(baseline.state_dict())
    tokens = torch.randn(1, 11, 16, device=device)
    distances = torch.rand(1, 11, 11, device=device)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(1, 11, dtype=torch.long, device=device)

    with torch.inference_mode():
        expected = baseline(tokens, distances, token_types, return_pair_rep=False)
        actual = stock(tokens, distances, token_types, return_pair_rep=False)

    torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
    assert "pair_rep" not in actual
    assert stock.gaussian.last_flagos_dispatch is None
    assert stock.encoder.layers[0].self_attn.last_flagos_dispatch is None
    assert stock.last_flagos_runtime_dispatch.active
    # The frozen baseline needs this temporary official FlagGems override for
    # Tesla V100 naming. It is restored after the scope exits.
    assert stock.last_flagos_runtime_dispatch.vendor_hint == "nvidia"
    assert "FLAGGEMS_VENDOR" not in os.environ
    assert current_flagos_runtime_dispatch() is None
