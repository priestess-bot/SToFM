import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "geneformer_001"))

from model.se2transformer import GaussianModule, MultiheadAttention, SToFMModel, TransformerEncoder
from model.flagos_backend import _operator_backend
from model.flagos_runtime import (
    STOFM_ATEN_ALLOWLIST,
    current_flagos_runtime_dispatch,
    flagos_inference_scope,
    validate_flagos_mode,
)
from model.utils import SToFMConfig


def _config(backend, attention_backend=None, mode="torch"):
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
        flagos_mode=mode,
        flagos_backend=backend,
        flagos_attention_backend=attention_backend,
    )


def test_config_preserves_torch_default_and_optimized_backend_selection():
    default = _config("flaggems")
    assert default.flagos_mode == "torch"
    config = _config("flaggems", mode="optimized")
    assert GaussianModule(config).flagos_backend == "flaggems"
    assert MultiheadAttention(config).flagos_backend == "flaggems"
    assert GaussianModule(config).flagos_mode == "optimized"
    assert MultiheadAttention(config).flagos_mode == "optimized"

    restored = SToFMConfig(**config.to_dict())
    assert restored.flagos_mode == "optimized"
    assert restored.flagos_backend == "flaggems"
    assert restored.flagos_attention_backend == "flaggems"

    overridden = _config("flaggems", attention_backend="torch", mode="optimized")
    assert MultiheadAttention(overridden).flagos_backend == "torch"


def test_explicit_target_backend_names_reach_the_public_bridge_without_vendor_imports():
    assert _operator_backend("ascend") == "ascend"
    assert _operator_backend("mthreads") == "mthreads"
    assert GaussianModule(_config("ascend", mode="optimized")).flagos_backend == "ascend"
    assert MultiheadAttention(_config("mthreads", mode="optimized")).flagos_backend == "mthreads"


def test_flagos_modes_and_torch_scope_are_validated_without_flaggems_imports():
    assert validate_flagos_mode("STOCK") == "stock"
    with pytest.raises(ValueError, match="flagos_mode"):
        validate_flagos_mode("invalid")
    assert current_flagos_runtime_dispatch() is None
    with flagos_inference_scope("torch") as dispatch:
        assert not dispatch.active
        assert dispatch.registered_aten_ops == ()
    assert current_flagos_runtime_dispatch() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems V100 integration requires CUDA")
def test_gaussian_adapter_matches_torch_fallback():
    device = torch.device("cuda")
    torch.manual_seed(13)
    baseline = GaussianModule(_config("torch")).to(device).eval()
    optimized = GaussianModule(_config("flaggems", mode="optimized")).to(device).eval()
    optimized.load_state_dict(baseline.state_dict())
    distances = torch.rand(2, 17, 17, device=device)
    distances[:, 0, 0] = 0.0

    with torch.inference_mode():
        expected = baseline(distances)
        actual = optimized(distances)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    assert optimized.last_flagos_dispatch is not None
    assert optimized.last_flagos_dispatch.selected == "inductor"
    assert optimized.last_flagos_dispatch.precision == "fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems V100 integration requires CUDA")
def test_attention_adapter_preserves_pair_state_and_weights():
    device = torch.device("cuda")
    torch.manual_seed(17)
    baseline = MultiheadAttention(_config("torch")).to(device).eval()
    optimized = MultiheadAttention(
        _config("torch", attention_backend="flaggems", mode="optimized")
    ).to(device).eval()
    optimized.load_state_dict(baseline.state_dict())
    query = torch.randn(11, 2, 32, device=device)
    bias = torch.randn(2, 4, 11, 11, device=device)
    padding = torch.zeros(2, 11, dtype=torch.bool, device=device)
    padding[0, -2:] = True

    with torch.inference_mode():
        expected = baseline(query, query, query, bias, key_padding_mask=padding, need_weights=True)
        actual = optimized(query, query, query, bias, key_padding_mask=padding, need_weights=True)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=3e-4, atol=3e-5)
    assert optimized.last_flagos_dispatch is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems V100 integration requires CUDA")
def test_encoder_can_skip_only_the_unused_final_pair_representation():
    device = torch.device("cuda")
    encoder = TransformerEncoder(_config("flaggems", mode="optimized")).to(device).eval()
    embeddings = torch.randn(2, 13, 32, device=device)
    token_types = torch.zeros(2, 13, dtype=torch.long, device=device)
    bias = torch.randn(2, 4, 13, 13, device=device)

    states, attentions, pair_rep = encoder(
        bias,
        embeddings,
        token_types,
        last_state_only=True,
        return_pair_rep=False,
    )
    assert len(states) == 1
    assert attentions == []
    assert pair_rep is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems V100 integration requires CUDA")
def test_end_to_end_adapter_preserves_input_gradients():
    device = torch.device("cuda")
    torch.manual_seed(23)
    baseline = SToFMModel(_config("torch")).to(device).eval()
    optimized = SToFMModel(_config("flaggems", mode="optimized")).to(device).eval()
    optimized.load_state_dict(baseline.state_dict())

    reference_tokens = torch.randn(1, 7, 16, device=device, requires_grad=True)
    optimized_tokens = reference_tokens.detach().clone().requires_grad_(True)
    reference_distances = torch.rand(1, 7, 7, device=device)
    reference_distances[:, 0, 0] = 0.0
    reference_distances.requires_grad_(True)
    optimized_distances = reference_distances.detach().clone().requires_grad_(True)
    token_types = torch.zeros(1, 7, dtype=torch.long, device=device)

    expected = baseline(reference_tokens, reference_distances, token_types, return_pair_rep=True)
    actual = optimized(optimized_tokens, optimized_distances, token_types, return_pair_rep=True)
    expected_loss = expected["last_hidden_state"].square().mean() + expected["pair_rep"].square().mean()
    actual_loss = actual["last_hidden_state"].square().mean() + actual["pair_rep"].square().mean()
    expected_grads = torch.autograd.grad(expected_loss, (reference_tokens, reference_distances))
    actual_grads = torch.autograd.grad(actual_loss, (optimized_tokens, optimized_distances))

    torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual["pair_rep"], expected["pair_rep"], rtol=3e-4, atol=3e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=3e-5)
    assert not optimized.last_flagos_runtime_dispatch.active
    assert optimized.gaussian.last_flagos_dispatch is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA native inference path requires CUDA")
def test_explicit_nvidia_inference_path_preserves_full_stofm_semantics():
    device = torch.device("cuda")
    torch.manual_seed(53)
    baseline = SToFMModel(_config("torch")).to(device).eval()
    default = SToFMModel(_config("flaggems", mode="optimized")).to(device).eval()
    native = SToFMModel(_config("nvidia", mode="optimized")).to(device).eval()
    default.load_state_dict(baseline.state_dict())
    native.load_state_dict(baseline.state_dict())
    token_embeddings = torch.randn(1, 17, 16, device=device)
    distances = torch.rand(1, 17, 17, device=device)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(1, 17, dtype=torch.long, device=device)

    with torch.inference_mode():
        expected = baseline(token_embeddings, distances, token_types, return_pair_rep=True)
        default_output = default(token_embeddings, distances, token_types, return_pair_rep=True)
        actual = native(token_embeddings, distances, token_types, return_pair_rep=True)
    torch.testing.assert_close(default_output["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(default_output["pair_rep"], expected["pair_rep"], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual["pair_rep"], expected["pair_rep"], rtol=3e-4, atol=3e-5)
    assert default.gaussian.last_flagos_dispatch is not None
    assert default.gaussian.last_flagos_dispatch.selected == "inductor"
    assert default.encoder.layers[0].self_attn.last_flagos_dispatch is not None
    assert default.encoder.layers[0].self_attn.last_flagos_dispatch.selected == "nvidia"
    assert native.gaussian.last_flagos_dispatch is not None
    assert native.gaussian.last_flagos_dispatch.selected == "nvidia"
    assert native.encoder.layers[0].self_attn.last_flagos_dispatch is not None
    assert native.encoder.layers[0].self_attn.last_flagos_dispatch.selected == "nvidia"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems stock scope requires CUDA")
def test_stock_and_optimized_modes_preserve_canonical_inference_semantics_and_scope():
    device = torch.device("cuda")
    torch.manual_seed(79)
    baseline = SToFMModel(_config("torch")).to(device).eval()
    stock = SToFMModel(_config("nvidia", mode="stock")).to(device).eval()
    optimized = SToFMModel(_config("flaggems", mode="optimized")).to(device).eval()
    stock.load_state_dict(baseline.state_dict())
    optimized.load_state_dict(baseline.state_dict())
    tokens = torch.randn(1, 13, 16, device=device)
    distances = torch.rand(1, 13, 13, device=device)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(1, 13, dtype=torch.long, device=device)

    with torch.inference_mode():
        expected = baseline(tokens, distances, token_types, return_pair_rep=False)
        with stock.flagos_inference_scope() as runtime:
            assert runtime.active
            assert set(STOFM_ATEN_ALLOWLIST).issubset(set(runtime.registered_aten_ops))
            actual_stock = stock(tokens, distances, token_types, return_pair_rep=False)
        actual_optimized = optimized(tokens, distances, token_types, return_pair_rep=False)
    assert current_flagos_runtime_dispatch() is None
    for actual in (actual_stock, actual_optimized):
        torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-4, atol=3e-5)
        assert "pair_rep" not in actual
    assert stock.gaussian.last_flagos_dispatch is None
    assert stock.encoder.layers[0].self_attn.last_flagos_dispatch is None
    assert optimized.gaussian.last_flagos_dispatch is not None
    assert optimized.last_flagos_runtime_dispatch.active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="V100 FP16 validation requires CUDA")
def test_optimized_fp16_inference_matches_torch_and_records_fp16_dispatch():
    device = torch.device("cuda")
    torch.manual_seed(83)
    baseline = SToFMModel(_config("torch")).to(device).half().eval()
    optimized = SToFMModel(_config("nvidia", mode="optimized")).to(device).half().eval()
    optimized.load_state_dict(baseline.state_dict())
    tokens = torch.randn(1, 11, 16, device=device, dtype=torch.float16)
    distances = torch.rand(1, 11, 11, device=device, dtype=torch.float16)
    distances[:, 0, 0] = 0.0
    token_types = torch.zeros(1, 11, dtype=torch.long, device=device)

    with torch.inference_mode():
        expected = baseline(tokens, distances, token_types, return_pair_rep=False)
        actual = optimized(tokens, distances, token_types, return_pair_rep=False)
    torch.testing.assert_close(actual["last_hidden_state"], expected["last_hidden_state"], rtol=3e-2, atol=3e-3)
    assert optimized.gaussian.last_flagos_dispatch.precision == "fp16"
    assert optimized.encoder.layers[0].self_attn.last_flagos_dispatch.precision == "fp16"
