"""End-to-end SToFM checks that run only on the configured MTT S4000 target."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


def _musa_is_available() -> bool:
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        return False
    return bool(getattr(torch, "musa", None) and torch.musa.is_available())


pytestmark = pytest.mark.skipif(not _musa_is_available(), reason="MTT S4000 MUSA runtime is required")


from model.se2transformer import SToFMModel  # noqa: E402
from model.utils import SToFMConfig  # noqa: E402


def _config(*, optimized: bool, aten_dispatch: bool = True) -> SToFMConfig:
    return SToFMConfig(
        num_hidden_layers=2,
        embedding_dim=32,
        ffn_embedding_dim=64,
        num_attention_heads=4,
        gaussian_hidden_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        input_dim=16,
        flagos_mode="optimized" if optimized else "torch",
        flagos_backend="mthreads" if optimized else "torch",
        flagos_attention_backend="mthreads" if optimized else "torch",
        flagos_aten_dispatch=aten_dispatch,
    )


def _library_path() -> Path:
    value = os.getenv("FLAGGEMS_STOFM_MUSA_LIBRARY")
    if not value:
        pytest.skip("set FLAGGEMS_STOFM_MUSA_LIBRARY to run S4000 target tests")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"MUSA extension does not exist: {path}")
    return path


@pytest.mark.parametrize("has_padding", [False, True])
def test_musa_native_stofm_matches_torch_and_records_dispatch(monkeypatch, has_padding):
    """Prove both unpadded and padded inference select the MUSA backend."""

    library = _library_path()
    monkeypatch.setenv("FLAGGEMS_STOFM_MUSA_LIBRARY", str(library))
    monkeypatch.setenv("FLAGGEMS_STOFM_ENABLE_MUSA_NATIVE", "1")
    monkeypatch.setenv("FLAGGEMS_STOFM_REQUIRE_MUSA_NATIVE", "1")
    torch.manual_seed(2201)
    device = torch.device("musa")
    baseline = SToFMModel(_config(optimized=False)).to(device).eval()
    optimized = SToFMModel(_config(optimized=True, aten_dispatch=True)).to(device).eval()
    optimized.load_state_dict(baseline.state_dict())
    tokens = torch.randn(1, 13, 16, device=device)
    distances = torch.rand(1, 13, 13, device=device)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(1, 13, dtype=torch.long, device=device)
    if has_padding:
        token_types[:, -2:] = 3

    with torch.inference_mode():
        expected = baseline(tokens, distances, token_types, return_pair_rep=True)
        actual = optimized(tokens, distances, token_types, return_pair_rep=True)
        torch.musa.synchronize()

    torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual["pair_rep"], expected["pair_rep"], rtol=3e-4, atol=3e-5)
    assert optimized.gaussian.last_flagos_dispatch is not None
    assert optimized.gaussian.last_flagos_dispatch.selected == "mthreads"
    assert optimized.encoder.layers[0].self_attn.last_flagos_dispatch is not None
    assert optimized.encoder.layers[0].self_attn.last_flagos_dispatch.selected == "mthreads"
    assert optimized.last_flagos_runtime_dispatch is not None
    assert not optimized.last_flagos_runtime_dispatch.active
    assert "MUSA native SToFM operators are active" in optimized.last_flagos_runtime_dispatch.reason

    from flag_gems.experimental_ops.stofm_backends import mthreads

    for operator in ("gaussian_pair_bias", "pair_score_epilogue"):
        status = mthreads.native_extension_status(operator)
        assert status["privateuse1_kernel_registered"]
        assert status["library_exists"]
