"""Shared portable Vision/KRONOS reference workload for R2 workers.

This module deliberately has no FlagGems import.  The frozen-stock worker can
therefore run it against the exact pre-R2 package, while the optimized worker
can use the same deterministic inputs and reference semantics.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "fp32" else torch.float16


def _tolerance(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float16:
        return {"rtol": 2e-2, "atol": 2e-3}
    return {"rtol": 3e-4, "atol": 3e-5}


def _inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    marker_ids = torch.arange(args.markers, device=device, dtype=torch.long).remainder(args.marker_vocab)
    marker_ids = marker_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    marker_padding = torch.zeros(args.batch_size, args.markers, dtype=torch.bool, device=device)
    padding_count = args.markers // 4
    if padding_count:
        marker_padding[:, -padding_count:] = True
    return {
        "patch_tokens": torch.randn(
            args.batch_size,
            args.markers,
            args.tokens_per_marker,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "marker_ids": marker_ids.masked_fill(marker_padding, -1),
        "marker_padding": marker_padding,
        "marker_weight": torch.randn(args.marker_vocab, args.embedding_dim, device=device, dtype=dtype),
        "position": torch.randn(args.tokens_per_marker, args.embedding_dim, device=device, dtype=dtype),
        "token": torch.randn(args.tokens_per_marker, args.embedding_dim, device=device, dtype=dtype),
        "packed_swiglu": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            2 * args.swiglu_hidden,
            device=device,
            dtype=dtype,
        ),
        "residual_input": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "residual": torch.randn(
            args.batch_size,
            args.swiglu_sequence,
            args.embedding_dim,
            device=device,
            dtype=dtype,
        ),
        "norm_weight": torch.randn(args.embedding_dim, device=device, dtype=dtype),
        "norm_bias": torch.randn(args.embedding_dim, device=device, dtype=dtype),
    }


def torch_marker_token_embed(tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Portable implementation of the KRONOS marker-token contract."""
    patch_tokens = tensors["patch_tokens"]
    padding = tensors["marker_padding"].to(torch.bool)
    safe_marker_ids = tensors["marker_ids"].masked_fill(padding, 0)
    marker_features = F.embedding(safe_marker_ids, tensors["marker_weight"]).unsqueeze(-2)
    output = patch_tokens + marker_features
    output = output + tensors["position"].view(1, 1, *tensors["position"].shape)
    output = output + tensors["token"].view(1, 1, *tensors["token"].shape)
    output = output.masked_fill(padding[:, :, None, None], 0.0)
    batch, markers, tokens, embedding_dim = output.shape
    output = output.reshape(batch, markers * tokens, embedding_dim)
    token_padding_mask = padding[:, :, None].expand(-1, -1, tokens).reshape(batch, markers * tokens)
    return output, token_padding_mask


def torch_swiglu(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    first, second = tensors["packed_swiglu"].chunk(2, dim=-1)
    return F.silu(first) * second


def torch_residual_layer_norm(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    input_tensor = tensors["residual_input"]
    return F.layer_norm(
        input_tensor + tensors["residual"],
        (input_tensor.shape[-1],),
        tensors["norm_weight"],
        tensors["norm_bias"],
        1e-5,
    )


def torch_invoke(operation: str, tensors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
    if operation == "marker_token_embed":
        return torch_marker_token_embed(tensors)
    if operation == "swiglu":
        return torch_swiglu(tensors), None
    if operation == "residual_layer_norm":
        return torch_residual_layer_norm(tensors), None
    raise ValueError(f"unsupported operation: {operation}")
