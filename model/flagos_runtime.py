"""Scoped FlagGems dispatch for reproducible SToFM inference and training.

The scope deliberately uses ``flag_gems.use_gems()`` rather than a global
registration.  It can therefore measure FlagOS lifecycle cost separately from
steady-state inference and it does not leave ATen overrides installed after an
extraction job returns.
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from threading import RLock
from typing import Any, Iterator, Mapping, Optional, Tuple

import torch


FLAGOS_MODES = ("torch", "stock", "optimized")
# These names are present in the frozen stock commit and in the optimized R2
# branch. Keep this intentionally narrow until a trace demonstrates that an
# additional FlagGems implementation is both exercised and correct.
STOFM_ATEN_ALLOWLIST = ("addmm", "baddbmm", "bmm", "softmax")
STOFM_GEMM_ATEN_OPS = ("mm", "addmm", "bmm", "baddbmm")
STOFM_GEMM_BACKENDS = ("triton", "vendor")

# These are function names from FlagGems' operator manifest (not necessarily
# the ATen schema spelling).  Keep the list explicit: registering all 700+
# available functions makes the first V100 training step compile unrelated
# kernels and obscures the actual SToFM training contract.
STOFM_TRAINING_ALLOWLIST = (
    "_fused_adamw_",
    "mm",
    "addmm",
    "baddbmm",
    "bmm",
    "sum",
    "sum_dim",
    "sum_dim_out",
    "mean",
    "mean_dim",
    "vector_norm",
    "layer_norm",
    "layer_norm_backward",
    "gelu",
    "gelu_backward",
    "threshold_backward",
    "embedding",
    "embedding_dense_backward",
    "softmax",
    "softmax_backward",
    "mul",
    "true_divide",
    "true_divide_",
    "exp",
    "abs",
    "sub",
    "add",
    "add_",
    "where_self",
    "masked_fill",
    "index",
    "index_select",
    "index_select_backward",
    "nonzero",
    "remainder",
    "neg",
    "clamp_min",
    "mse_loss",
    "mse_loss_backward",
    "arange",
    "sqrt",
    "reciprocal",
    "copy_",
    "zero_",
    "ones_like",
    "zeros_like",
    "scalar_tensor",
    "addcdiv",
    "addcdiv_",
    "addcmul",
    "addcmul_",
    "lerp_scalar",
    "lerp_tensor",
    "lerp_scalar_",
    "lerp_tensor_",
    "sqrt_",
    "mul_",
    "sub_",
    "clamp_",
    "eq",
    "eq_scalar",
    "ne",
    "ne_scalar",
    "ge",
    "ge_scalar",
    "gt",
    "gt_scalar",
    "cat",
    "cumsum",
    "zeros",
    "broadcast_tensors",
    "broadcast_to",
    "narrow",
    "dropout",
    "square",
    "rsub_tensor",
    "sgn",
    "to_copy",
    "masked_fill_",
    "embedding_backward",
)

# View/metadata operations are recorded for completeness but do not represent
# a missing compute kernel.  The strict training audit excludes these names.
STOFM_TRAINING_METADATA_OPS = (
    "view",
    "reshape",
    "transpose",
    "t",
    "as_strided",
    "expand",
    "permute",
    "contiguous",
    "clone",
    "empty",
    "empty_like",
    "empty_strided",
    "resize_",
    "select",
    "slice",
    "detach",
    "_reshape_alias",
    "_to_copy",
    "_unsafe_view",
    "expand_as",
    "new_empty",
    "new_empty_strided",
    "new_zeros",
    "lift_fresh",
    "item",
    "_local_scalar_dense",
    "result_type",
    "squeeze",
    "unsqueeze",
    "to",
    "narrow",
    "linear",
    "matmul",
    "rsub",
    "square",
    "fill_",
    "set_",
    "detach_",
)


@dataclass(frozen=True)
class FlagOSRuntimeDispatch:
    """The actual scoped ATen dispatch state for one model invocation."""

    mode: str
    active: bool
    registered_aten_ops: Tuple[str, ...]
    reason: str
    vendor_hint: Optional[str] = None
    phase: str = "inference"
    strict: bool = False
    gemm_backend: str = "triton"
    vendor_gemm_library: Optional[str] = None


_ACTIVE_SCOPE: ContextVar[Optional[FlagOSRuntimeDispatch]] = ContextVar(
    "stofm_flagos_active_scope", default=None
)
_FLAGGEMS_SCOPE_LOCK = RLock()


def validate_flagos_mode(mode: str) -> str:
    normalized = mode.lower()
    if normalized not in FLAGOS_MODES:
        choices = ", ".join(FLAGOS_MODES)
        raise ValueError(f"flagos_mode must be one of: {choices}")
    return normalized


def validate_flagos_gemm_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized not in STOFM_GEMM_BACKENDS:
        choices = ", ".join(STOFM_GEMM_BACKENDS)
        raise ValueError(f"flagos_gemm_backend must be one of: {choices}")
    return normalized


def prepare_flagos_vendor_training_fusion(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> Optional[float]:
    """Precompile the NVIDIA Gaussian backward while only Vendor dispatch is active.

    The broad FlagGems ATen registrar is intentionally not entered here.  This
    helper is a boundary for benchmark/train drivers: it returns the one-time
    preparation cost in milliseconds and leaves the compiled callable cached in
    FlagGems' NVIDIA backend for the subsequent full training scope.
    """
    dispatch = getattr(
        model,
        "flagos_gemm_backend",
        getattr(getattr(model, "model", None), "flagos_gemm_backend", "triton"),
    )
    distances = batch.get("attn_bias")
    gaussian = getattr(getattr(model, "model", None), "gaussian", None)
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    attention = (
        encoder.layers[0].self_attn
        if encoder is not None and len(encoder.layers) > 0
        else None
    )
    gaussian_native = (
        gaussian is not None
        and getattr(gaussian, "flagos_training_implementation", "reference") == "native"
    )
    pair_native = (
        attention is not None
        and getattr(attention, "flagos_training_implementation", "reference") == "native"
    )
    if dispatch != "vendor" or not (gaussian_native or pair_native):
        return None
    if distances is None or gaussian is None or distances.device.type != "cuda":
        return None
    if distances.dtype != torch.float32:
        return None
    from flag_gems.experimental_ops.stofm_backends import nvidia
    from flag_gems.experimental_ops.vendor_gemm import vendor_gemm_scope

    batch_size, nodes, _ = distances.shape
    grad_output = torch.empty(
        (batch_size, gaussian.num_heads, nodes, nodes),
        device=distances.device,
        dtype=distances.dtype,
    )
    args = (
        grad_output,
        distances,
        gaussian.linear.weight,
        gaussian.linear.bias,
        gaussian.means.weight,
        gaussian.stds.weight,
        gaussian.proj[0].weight,
        gaussian.proj[0].bias,
        gaussian.proj[2].weight,
        distances.eq(0.0),
    )
    query_nodes = nodes
    head_dim = gaussian.num_heads and model.config.embedding_dim // gaussian.num_heads
    pair_args = None
    if pair_native:
        pair_query = torch.empty(
            (batch_size, gaussian.num_heads, query_nodes, head_dim),
            device=distances.device,
            dtype=distances.dtype,
        )
        pair_key = torch.empty_like(pair_query)
        pair_value = torch.empty_like(pair_query)
        pair_probabilities = torch.empty(
            (batch_size, gaussian.num_heads, query_nodes, nodes),
            device=distances.device,
            dtype=distances.dtype,
        )
        pair_grad_context = torch.empty_like(pair_query)
        pair_grad_bias = torch.empty_like(pair_probabilities)
        pair_padding = batch.get("token_types")
        pair_padding = (
            pair_padding.eq(getattr(model.config, "pad_type_id", 3))
            if pair_padding is not None
            else None
        )
        pair_args = (
            pair_grad_context,
            pair_grad_bias,
            None,
            pair_query,
            pair_key,
            pair_value,
            pair_probabilities,
            pair_padding,
            head_dim**-0.5,
            True,
            False,
            pair_padding is not None,
        )
    with torch.no_grad(), vendor_gemm_scope(required=True):
        gaussian_ms = (
            nvidia.prepare_gaussian_pair_bias_training_backward(*args)
            if gaussian_native
            else 0.0
        )
        pair_ms = (
            nvidia.prepare_pair_attention_training_backward(*pair_args)
            if pair_args is not None
            else 0.0
        )
    return float(gaussian_ms + pair_ms)


def current_flagos_runtime_dispatch() -> Optional[FlagOSRuntimeDispatch]:
    """Return the current scoped dispatch record, if an ATen scope is active."""
    return _ACTIVE_SCOPE.get()


def current_flagos_training_dispatch() -> Optional[FlagOSRuntimeDispatch]:
    """Return the active training scope, if the caller is inside one."""
    record = _ACTIVE_SCOPE.get()
    if record is not None and record.phase == "training":
        return record
    return None


def _registered_aten_ops(flag_gems) -> Tuple[str, ...]:
    try:
        registered = flag_gems.all_registered_ops()
    except (AttributeError, NameError):
        return ()
    return tuple(sorted(str(item) for item in registered))


def _enable_musa_stofm_minimal_import() -> bool:
    """Request the explicit MUSA native-only FlagGems surface when available."""

    try:
        import torch_musa  # noqa: F401
    except ImportError:
        return False
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        return False
    os.environ.setdefault("FLAGGEMS_STOFM_MUSA_MINIMAL_IMPORT", "1")
    return True


@contextmanager
def _temporary_vendor_hint() -> Iterator[Optional[str]]:
    """Work around frozen FlagGems V100 detection without mutating its source.

    The frozen stock commit identifies NVIDIA only when a device name contains
    the literal word ``NVIDIA``. V100 reports ``Tesla V100-SXM2-16GB`` instead.
    The official FlagGems vendor override is scoped around import/registration
    and restored immediately, so the immutable stock package remains intact.
    """
    vendor_keys = ("GEMS_VENDOR", "FLAGGEMS_VENDOR", "GEMS_BACKEND", "FLAGGEMS_BACKEND")
    if any(key in os.environ for key in vendor_keys) or not torch.cuda.is_available():
        yield None
        return
    os.environ["FLAGGEMS_VENDOR"] = "nvidia"
    try:
        yield "nvidia"
    finally:
        os.environ.pop("FLAGGEMS_VENDOR", None)


@contextmanager
def flagos_inference_scope(
    mode: str,
    *,
    enabled: bool = True,
    disabled_reason: Optional[str] = None,
) -> Iterator[FlagOSRuntimeDispatch]:
    """Install a temporary FlagGems ATen dispatch scope for inference only.

    ``stock`` and ``optimized`` use the same ATen allowlist.  The distinction
    is intentional: only ``optimized`` permits SToFM's versioned composite API
    at the model boundary.  This makes the frozen-stock baseline a real
    FlagGems inference run without accidentally loading R2 experimental code.
    """
    normalized = validate_flagos_mode(mode)
    if normalized == "torch":
        yield FlagOSRuntimeDispatch(
            mode=normalized,
            active=False,
            registered_aten_ops=(),
            reason="Torch baseline mode",
        )
        return
    if not enabled:
        yield FlagOSRuntimeDispatch(
            mode=normalized,
            active=False,
            registered_aten_ops=(),
            reason=disabled_reason or "FlagOS dispatch is inference-only; training or autograd is active",
        )
        return

    active = _ACTIVE_SCOPE.get()
    if active is not None:
        if active.mode != normalized:
            raise RuntimeError(
                f"cannot nest FlagOS mode '{normalized}' inside active mode '{active.mode}'"
            )
        yield active
        return

    # FlagGems' registrar owns a process-global ATen Library. Serializing
    # outer scopes prevents two Python threads from destroying each other's
    # temporary registration while retaining re-entrant use within one scope.
    with _FLAGGEMS_SCOPE_LOCK:
        with _temporary_vendor_hint() as vendor_hint:
            if normalized == "optimized":
                _enable_musa_stofm_minimal_import()
            try:
                import flag_gems
            except ImportError as exc:
                raise RuntimeError(
                    f"flagos_mode='{normalized}' requires the pinned FlagGems package"
                ) from exc
            if getattr(flag_gems, "MUSA_STOFM_MINIMAL_RUNTIME", False):
                yield FlagOSRuntimeDispatch(
                    mode=normalized,
                    active=False,
                    registered_aten_ops=(),
                    reason=(
                        "MUSA native SToFM operators are active; FlagGems global "
                        "ATen dispatch is unavailable because this runtime has no Triton driver"
                    ),
                    vendor_hint=vendor_hint,
                )
                return
            with flag_gems.use_gems(include=STOFM_ATEN_ALLOWLIST):
                record = FlagOSRuntimeDispatch(
                    mode=normalized,
                    active=True,
                    registered_aten_ops=_registered_aten_ops(flag_gems),
                    reason="scoped FlagGems ATen dispatch",
                    vendor_hint=vendor_hint,
                    phase="inference",
                    strict=False,
                )
                token = _ACTIVE_SCOPE.set(record)
                try:
                    yield record
                finally:
                    _ACTIVE_SCOPE.reset(token)


@contextmanager
def flagos_training_scope(
    mode: str = "optimized",
    *,
    enabled: bool = True,
    strict: bool = True,
    include: Optional[Tuple[str, ...]] = None,
    record: bool = False,
    record_path: Optional[str] = None,
    gemm_backend: str = "triton",
) -> Iterator[FlagOSRuntimeDispatch]:
    """Keep a curated FlagGems registration alive for a complete train step.

    Unlike :func:`flagos_inference_scope`, this context is deliberately owned
    by the training loop and must surround ``forward``, ``backward`` and
    ``optimizer.step``.  The model only observes the context; it does not open
    a nested scope that would disappear before autograd runs.
    """
    normalized = validate_flagos_mode(mode)
    gemm_backend = validate_flagos_gemm_backend(gemm_backend)
    requested = (
        tuple(STOFM_TRAINING_ALLOWLIST) if include is None else tuple(include)
    )
    if normalized == "torch":
        yield FlagOSRuntimeDispatch(
            mode=normalized,
            active=False,
            registered_aten_ops=(),
            reason="Torch baseline mode",
            phase="training",
            strict=strict,
            gemm_backend=gemm_backend,
        )
        return
    if not enabled:
        yield FlagOSRuntimeDispatch(
            mode=normalized,
            active=False,
            registered_aten_ops=(),
            reason="FlagOS training dispatch is disabled",
            phase="training",
            strict=strict,
            gemm_backend=gemm_backend,
        )
        return

    active = _ACTIVE_SCOPE.get()
    if active is not None:
        if active.mode != normalized or active.phase != "training":
            raise RuntimeError(
                f"cannot nest FlagOS training mode '{normalized}' inside "
                f"active {active.phase} mode '{active.mode}'"
            )
        yield active
        return

    with _FLAGGEMS_SCOPE_LOCK:
        with _temporary_vendor_hint() as vendor_hint:
            if normalized == "optimized":
                _enable_musa_stofm_minimal_import()
            try:
                import flag_gems
            except ImportError as exc:
                raise RuntimeError(
                    "FlagOS training requires the pinned FlagGems package"
                ) from exc
            if getattr(flag_gems, "MUSA_STOFM_MINIMAL_RUNTIME", False):
                reason = (
                    "MUSA native SToFM runtime has no general FlagGems ATen "
                    "training driver"
                )
                if strict:
                    raise RuntimeError(reason)
                yield FlagOSRuntimeDispatch(
                    mode=normalized,
                    active=False,
                    registered_aten_ops=(),
                    reason=reason,
                    vendor_hint=vendor_hint,
                    phase="training",
                    strict=strict,
                )
                return

            # Vendor GEMM is a separate FlagOS dispatcher. Remove its ATen
            # names from the Python Triton registrar so the two scopes never
            # override each other; the strict record adds them back as owned
            # FlagOS implementations below.
            requested_flaggems = tuple(
                name for name in requested
                if not (gemm_backend == "vendor" and name in STOFM_GEMM_ATEN_OPS)
            )
            # ``use_gems`` accepts manifest function names such as
            # ``layer_norm`` and ``vector_norm``; it expands them to the
            # corresponding native_layer_norm/linalg_vector_norm schemas.
            with ExitStack() as stack:
                vendor_dispatch = None
                if gemm_backend == "vendor":
                    from flag_gems.experimental_ops.vendor_gemm import vendor_gemm_scope

                    vendor_dispatch = stack.enter_context(vendor_gemm_scope(required=True))
                if requested_flaggems:
                    stack.enter_context(
                        flag_gems.use_gems(
                            include=requested_flaggems,
                            record=record,
                            path=record_path,
                        )
                    )
                registered = set(_registered_aten_ops(flag_gems))
                if vendor_dispatch is not None:
                    registered.update(STOFM_GEMM_ATEN_OPS)
                registered = tuple(sorted(registered))
                missing = sorted(set(requested_flaggems) - set(registered))
                if strict and missing:
                    raise RuntimeError(
                        "FlagOS training requested operators that FlagGems did not "
                        "register: " + ", ".join(missing)
                    )
                record = FlagOSRuntimeDispatch(
                    mode=normalized,
                    active=True,
                    registered_aten_ops=registered,
                    reason="scoped FlagGems ATen training dispatch",
                    vendor_hint=vendor_hint,
                    phase="training",
                    strict=strict,
                    gemm_backend=gemm_backend,
                    vendor_gemm_library=(
                        vendor_dispatch.loaded_library if vendor_dispatch is not None else None
                    ),
                )
                token = _ACTIVE_SCOPE.set(record)
                try:
                    yield record
                finally:
                    _ACTIVE_SCOPE.reset(token)
