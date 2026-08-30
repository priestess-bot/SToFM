import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.train_stofm_fake_flagos import (
    FakeTrainingConfig,
    SyntheticSToFMDataset,
    _flaggems_log_summary,
    _kernel_evidence,
    _parameter_sha256,
)
from model.se2transformer import SToFMForMaskedLM
from model.utils import SToFMConfig


def _small_config(**overrides):
    values = dict(
        num_hidden_layers=1,
        input_dim=8,
        embedding_dim=16,
        ffn_embedding_dim=16,
        num_attention_heads=4,
        gaussian_hidden_dim=8,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        flagos_mode="torch",
        flagos_backend="torch",
        flagos_attention_backend="torch",
    )
    values.update(overrides)
    return SToFMConfig(**values)


def test_synthetic_dataset_is_reproducible_and_matches_stofm_contract():
    config = FakeTrainingConfig(seed=123, batch_size=2, nodes=9, input_dim=8, embedding_dim=16)
    first = SyntheticSToFMDataset(config, torch.device("cpu"))
    second = SyntheticSToFMDataset(config, torch.device("cpu"))
    for name in ("token_embeddings", "attn_bias", "token_types", "labels", "pair_labels"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))
    assert first.token_embeddings.shape == (2, 9, 8)
    assert first.attn_bias.shape == (2, 9, 9)
    assert first.labels.shape == (2, 9, 16)
    assert first.pair_labels.shape == (2, 9, 9)
    assert torch.all(first.token_types[:, 0] == 1)
    assert torch.all(first.token_types[:, -1] == 3)
    assert torch.all(first.labels[:, 0] == -100)
    assert torch.all(first.pair_labels[:, 0, :] == -100)


def test_masked_losses_are_finite_and_update_parameters_on_cpu():
    torch.manual_seed(19)
    model = SToFMForMaskedLM(_small_config()).train()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(seed=19, batch_size=1, nodes=7, input_dim=8, embedding_dim=16),
        torch.device("cpu"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    before = _parameter_sha256(model)
    output = model(**dataset.batch())
    loss = output["loss"] + output["pair_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    assert _parameter_sha256(model) != before


def test_registered_stofm_composites_have_input_gradients_on_cpu():
    torch.manual_seed(23)
    config = _small_config(
        flagos_mode="optimized",
        flagos_backend="nvidia",
        flagos_attention_backend="nvidia",
    )
    model = SToFMForMaskedLM(config).train()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(seed=23, batch_size=1, nodes=6, input_dim=8, embedding_dim=16),
        torch.device("cpu"),
    )
    batch = dataset.batch()
    for tensor_name in ("token_embeddings", "attn_bias"):
        batch[tensor_name] = batch[tensor_name].requires_grad_(True)
    output = model(**batch)
    loss = output["last_hidden_state"].square().mean() + output["pair_loss"]
    grads = torch.autograd.grad(loss, (batch["token_embeddings"], batch["attn_bias"]))
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads)


def test_masked_cosine_reduction_matches_pytorch_cosine_embedding_loss():
    torch.manual_seed(29)
    prediction = torch.randn(2, 5, 16)
    target = torch.randn(2, 5, 16)
    valid = torch.tensor([[True, True, False, True, False], [True, False, True, True, True]])
    normalized_prediction = torch.nn.functional.normalize(prediction, dim=-1)
    normalized_target = torch.nn.functional.normalize(target, dim=-1)
    ours_per = 1.0 - (normalized_prediction * normalized_target).sum(dim=-1)
    ours = (ours_per * valid).sum() / valid.sum()
    reference = torch.nn.functional.cosine_embedding_loss(
        prediction[valid],
        target[valid],
        torch.ones(int(valid.sum())),
        reduction="mean",
    )
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-6)


def test_masked_pair_mse_reduction_matches_pytorch_mse_loss():
    torch.manual_seed(30)
    prediction = torch.randn(2, 5, 5)
    target = torch.randn(2, 5, 5)
    valid = torch.tensor(
        [
            [
                [True, True, False, True, False],
                [True, True, True, False, True],
                [False, True, True, True, False],
                [True, False, True, True, True],
                [False, True, False, True, True],
            ],
            [
                [True, False, True, True, True],
                [False, True, True, False, True],
                [True, True, False, True, False],
                [True, False, True, True, True],
                [True, True, True, False, True],
            ],
        ]
    )
    delta = prediction - torch.where(valid, target, torch.zeros_like(target))
    ours = (delta * delta * valid).sum() / valid.sum()
    reference = torch.nn.functional.mse_loss(prediction[valid], target[valid])
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-6)


def test_model_mcm_and_pdr_losses_match_original_pytorch_objectives():
    torch.manual_seed(32)
    config = _small_config()
    model = SToFMForMaskedLM(config).eval()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(seed=32, batch_size=1, nodes=7, input_dim=8, embedding_dim=16),
        torch.device("cpu"),
    )
    batch = dataset.batch()
    outputs = model(**batch)

    valid = batch["labels"][:, :, 0].ne(-100.0)
    raw_prediction = model.lm_head(outputs["last_hidden_state"])
    reference_mcm = torch.nn.functional.cosine_embedding_loss(
        torch.nn.functional.normalize(raw_prediction[valid], dim=-1),
        batch["labels"][valid],
        torch.ones(int(valid.sum())),
        reduction="mean",
    )
    pair_valid = batch["pair_labels"].ne(-100.0)
    raw_pair_prediction = model.pair_head(outputs["pair_rep"]).squeeze(-1)
    reference_pdr = torch.nn.functional.mse_loss(
        raw_pair_prediction[pair_valid], batch["pair_labels"][pair_valid], reduction="mean"
    )
    torch.testing.assert_close(outputs["loss"], reference_mcm, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs["pair_loss"], reference_pdr, rtol=1e-6, atol=1e-6)


def test_checkpoint_payload_restores_model_and_optimizer_state(tmp_path):
    torch.manual_seed(41)
    config = _small_config()
    model = SToFMForMaskedLM(config).train()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(seed=41, batch_size=1, nodes=6, input_dim=8, embedding_dim=16),
        torch.device("cpu"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**dataset.batch())
    (outputs["loss"] + outputs["pair_loss"]).backward()
    optimizer.step()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 7,
            "config": config.to_dict(),
        },
        checkpoint,
    )
    restored = SToFMForMaskedLM(config).train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False, fused=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    assert payload["step"] == 7
    assert _parameter_sha256(restored) == _parameter_sha256(model)


def test_profile_evidence_helpers_parse_kernel_and_flaggems_logs(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        '{"traceEvents": ['
        '{"ph":"X","cat":"kernel","name":"mm_kernel_general"},'
        '{"ph":"X","cat":"kernel","name":"void at::native::sum_kernel"},'
        '{"ph":"X","cat":"kernel","name":"mm_kernel_general"},'
        '{"ph":"X","cat":"cpu_op","name":"aten::mm"}'
        ']}',
        encoding="utf-8",
    )
    evidence = _kernel_evidence(trace)
    assert evidence["event_count"] == 3
    assert evidence["kernel_counts"]["mm_kernel_general"] == 2
    assert evidence["candidate_flaggems_kernel_names"] == ["mm_kernel_general"]

    log = tmp_path / "flaggems_ops.log"
    log.write_text(
        "[DEBUG] flag_gems.ops.mm.mm: GEMS MM\n"
        "[DEBUG] flag_gems.ops.softmax.softmax: GEMS SOFTMAX\n"
        "[DEBUG] flag_gems.ops.mm.mm: GEMS MM\n",
        encoding="utf-8",
    )
    summary = _flaggems_log_summary(log)
    assert summary["line_count"] == 3
    assert summary["operator_functions"] == ["mm", "softmax"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems training integration requires CUDA")
@pytest.mark.skipif(
    not bool(__import__("os").environ.get("RUN_FLAGOS_TRAINING_INTEGRATION")),
    reason="set RUN_FLAGOS_TRAINING_INTEGRATION=1 for the V100 integration test",
)
def test_flagos_training_scope_runs_a_real_forward_backward_step():
    from model.flagos_runtime import flagos_training_scope

    torch.manual_seed(31)
    config = _small_config(
        flagos_mode="optimized",
        flagos_backend="nvidia",
        flagos_attention_backend="nvidia",
    )
    model = SToFMForMaskedLM(config).cuda().train()
    dataset = SyntheticSToFMDataset(
        FakeTrainingConfig(seed=31, batch_size=1, nodes=6, input_dim=8, embedding_dim=16),
        torch.device("cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    with flagos_training_scope(strict=False) as dispatch:
        optimizer.zero_grad(set_to_none=True)
        output = model(**dataset.batch())
        loss = output["loss"] + output["pair_loss"]
        loss.backward()
        optimizer.step()
    assert dispatch.active
    assert model.model.last_flagos_runtime_dispatch.phase == "training"
    assert model.model.gaussian.last_flagos_dispatch is not None
    assert model.model.encoder.layers[0].self_attn.last_flagos_dispatch is not None
    assert torch.isfinite(loss)
