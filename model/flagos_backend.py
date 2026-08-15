"""Optional, versioned bridge from SToFM to FlagGems experimental operators."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


STOFM_FLAGGEMS_API_VERSION = 2


@dataclass(frozen=True)
class FlagOSDispatch:
    operator: str
    requested: str
    selected: str
    precision: str
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
    if backend not in {"torch", "auto", "flaggems", "inductor", "nvidia", "ascend", "mthreads"}:
        raise ValueError(
            "flagos_backend must be one of: torch, auto, flaggems, inductor, nvidia, ascend, mthreads"
        )
    if backend == "torch":
        return None
    return _load_experimental_ops(required=backend in {"flaggems", "inductor", "nvidia", "ascend", "mthreads"})


def _operator_backend(backend: str) -> str:
    return backend if backend in {"inductor", "nvidia", "ascend", "mthreads"} else "auto"


def _dispatch_from_public(dispatch) -> FlagOSDispatch:
    return FlagOSDispatch(
        operator=dispatch.operator,
        requested=dispatch.requested,
        selected=dispatch.selected,
        precision=dispatch.precision,
        reason=dispatch.reason,
    )


def gaussian_pair_bias(module, distances: torch.Tensor, backend: str) -> Optional[Tuple[torch.Tensor, FlagOSDispatch]]:
    """Return ``None`` only for the optional auto fallback path."""
    ops = _get_ops(backend)
    if ops is None:
        return None
    operator_backend = _operator_backend(backend)
    resolution = ops.resolve_stofm_backend(distances, operator_backend)
    if backend in {"flaggems", "inductor", "nvidia", "ascend", "mthreads"} and resolution.selected == "torch":
        raise RuntimeError(f"FlagGems has no accelerated backend for this input: {resolution.reason}")
    output, dispatch = ops.stofm_gaussian_pair_bias(
        distances,
        module.linear.weight,
        module.linear.bias,
        module.means.weight,
        module.stds.weight,
        module.proj[0].weight,
        module.proj[0].bias,
        module.proj[2].weight,
        module.proj[2].bias,
        backend=operator_backend,
        return_dispatch=True,
    )
    dispatch = _dispatch_from_public(dispatch)
    if backend in {"flaggems", "inductor", "nvidia", "ascend", "mthreads"} and dispatch.selected == "torch":
        raise RuntimeError(f"FlagGems fell back to Torch for Gaussian pair bias: {dispatch.reason}")
    return output, dispatch


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
    operator_backend = _operator_backend(backend)
    # The V100-selected O5 configuration combines the auto/Inductor Gaussian
    # path with the native NVIDIA pair-score epilogue. The epilogue itself
    # retains its reference fallback for autograd, dropout, and unsupported
    # layouts; non-CUDA targets keep their normal auto resolution.
    if backend in {"auto", "flaggems"} and query.device.type == "cuda":
        operator_backend = "nvidia"
    resolution = ops.resolve_stofm_backend(query, operator_backend)
    if backend in {"flaggems", "inductor", "nvidia", "ascend", "mthreads"} and resolution.selected == "torch":
        raise RuntimeError(f"FlagGems has no accelerated backend for this input: {resolution.reason}")
    result, dispatch = ops.stofm_pair_attention(
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
        backend=operator_backend,
        return_dispatch=True,
    )
    dispatch = _dispatch_from_public(dispatch)
    if backend in {"flaggems", "inductor", "nvidia", "ascend", "mthreads"} and dispatch.selected == "torch":
        raise RuntimeError(f"FlagGems fell back to Torch for pair attention: {dispatch.reason}")
    return result, dispatch
