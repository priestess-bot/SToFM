"""Optional, versioned bridge from SToFM to FlagGems experimental operators."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


STOFM_FLAGGEMS_API_VERSION = 1


@dataclass(frozen=True)
class FlagOSDispatch:
    selected: str
    reason: str


def _load_experimental_ops(required: bool):
    try:
        from flag_gems import experimental_ops
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "flagos_backend='flaggems' requires the pinned FlagGems experimental package"
            ) from exc
        return None
    if getattr(experimental_ops, "STOFM_EXPERIMENTAL_API_VERSION", None) != STOFM_FLAGGEMS_API_VERSION:
        if required:
            raise RuntimeError(
                "The installed FlagGems package does not implement the required SToFM experimental API"
            )
        return None
    return experimental_ops


def _get_ops(backend: str):
    if backend not in {"torch", "auto", "flaggems"}:
        raise ValueError("flagos_backend must be one of: torch, auto, flaggems")
    if backend == "torch":
        return None
    return _load_experimental_ops(required=backend == "flaggems")


def gaussian_pair_bias(module, distances: torch.Tensor, backend: str) -> Optional[Tuple[torch.Tensor, FlagOSDispatch]]:
    """Return ``None`` only for the optional auto fallback path."""
    ops = _get_ops(backend)
    if ops is None:
        return None
    resolution = ops.resolve_stofm_backend(distances, "auto")
    if backend == "flaggems" and resolution.selected == "torch":
        raise RuntimeError(f"FlagGems has no accelerated backend for this input: {resolution.reason}")
    output = ops.stofm_gaussian_pair_bias(
        distances,
        module.linear.weight,
        module.linear.bias,
        module.means.weight,
        module.stds.weight,
        module.proj[0].weight,
        module.proj[0].bias,
        module.proj[2].weight,
        module.proj[2].bias,
        backend="auto",
    )
    return output, FlagOSDispatch(resolution.selected, resolution.reason)


def pair_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair_bias: torch.Tensor,
    *,
    key_padding_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    return_pair: bool,
    return_weights: bool,
    backend: str,
):
    """Run pair-state attention through the public FlagGems direct API."""
    ops = _get_ops(backend)
    if ops is None:
        return None
    resolution = ops.resolve_stofm_backend(query, "auto")
    if backend == "flaggems" and resolution.selected == "torch":
        raise RuntimeError(f"FlagGems has no accelerated backend for this input: {resolution.reason}")
    result = ops.stofm_pair_attention(
        query,
        key,
        value,
        pair_bias,
        key_padding_mask=key_padding_mask,
        dropout_p=dropout_p,
        training=training,
        # SToFM scales q before this bridge, so the score alpha is one.
        scale=1.0,
        return_pair=return_pair,
        return_weights=return_weights,
        assume_finite_pair_bias=key_padding_mask is None,
        backend="auto",
    )
    return result, FlagOSDispatch(resolution.selected, resolution.reason)
