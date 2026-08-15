"""CPU checks for the frozen Vision F0 worker's portable boundary semantics."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from vision_r2_common import _inputs
from vision_r2_v100_stock_worker import STOCK_STAGES, _invoke


def _args():
    return argparse.Namespace(
        batch_size=1,
        markers=4,
        tokens_per_marker=3,
        embedding_dim=5,
        marker_vocab=7,
        swiglu_sequence=2,
        swiglu_hidden=6,
    )


def test_frozen_stock_worker_uses_the_documented_portable_boundaries():
    torch.manual_seed(19)
    tensors = _inputs(_args(), torch.device("cpu"), torch.float32)

    marker, marker_dispatch, mask = _invoke("marker_token_embed", tensors)
    padding = tensors["marker_padding"]
    safe_ids = tensors["marker_ids"].masked_fill(padding, 0)
    expected_marker = tensors["patch_tokens"] + F.embedding(safe_ids, tensors["marker_weight"]).unsqueeze(-2)
    expected_marker = expected_marker + tensors["position"].view(1, 1, 3, 5)
    expected_marker = expected_marker + tensors["token"].view(1, 1, 3, 5)
    expected_marker = expected_marker.masked_fill(padding[:, :, None, None], 0.0).reshape(1, 12, 5)
    torch.testing.assert_close(marker, expected_marker)
    assert torch.equal(mask, padding[:, :, None].expand(-1, -1, 3).reshape(1, 12))
    assert marker_dispatch.requested == "stock"
    assert marker_dispatch.selected == "torch"

    swiglu, swiglu_dispatch, no_mask = _invoke("swiglu", tensors)
    first, second = tensors["packed_swiglu"].chunk(2, dim=-1)
    torch.testing.assert_close(swiglu, F.silu(first) * second)
    assert no_mask is None
    assert swiglu_dispatch.precision == "fp32"

    layer_norm, layer_norm_dispatch, no_mask = _invoke("residual_layer_norm", tensors)
    expected_layer_norm = F.layer_norm(
        tensors["residual_input"] + tensors["residual"],
        (5,),
        tensors["norm_weight"],
        tensors["norm_bias"],
        1e-5,
    )
    torch.testing.assert_close(layer_norm, expected_layer_norm)
    assert no_mask is None
    assert layer_norm_dispatch.operator == "residual_layer_norm"


def test_every_frozen_vision_boundary_has_a_distinct_stock_stage():
    assert [stage.name for stage in STOCK_STAGES] == [
        "V0s_marker_token_stock_flagos",
        "V2s_swiglu_stock_flagos",
        "V4s_residual_layer_norm_stock_flagos",
    ]
