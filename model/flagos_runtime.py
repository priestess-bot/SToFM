"""Scoped FlagGems dispatch for reproducible SToFM inference.

The scope deliberately uses ``flag_gems.use_gems()`` rather than a global
registration.  It can therefore measure FlagOS lifecycle cost separately from
steady-state inference and it does not leave ATen overrides installed after an
extraction job returns.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from threading import RLock
from typing import Iterator, Optional, Tuple

import torch


FLAGOS_MODES = ("torch", "stock", "optimized")
# These names are present in the frozen stock commit and in the optimized R2
# branch. Keep this intentionally narrow until a trace demonstrates that an
# additional FlagGems implementation is both exercised and correct.
STOFM_ATEN_ALLOWLIST = ("addmm", "baddbmm", "bmm", "softmax")


@dataclass(frozen=True)
class FlagOSRuntimeDispatch:
    """The actual scoped ATen dispatch state for one model invocation."""

    mode: str
    active: bool
    registered_aten_ops: Tuple[str, ...]
    reason: str
    vendor_hint: Optional[str] = None


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


def current_flagos_runtime_dispatch() -> Optional[FlagOSRuntimeDispatch]:
    """Return the current scoped dispatch record, if an ATen scope is active."""
    return _ACTIVE_SCOPE.get()


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
                )
                token = _ACTIVE_SCOPE.set(record)
                try:
                    yield record
                finally:
                    _ACTIVE_SCOPE.reset(token)
