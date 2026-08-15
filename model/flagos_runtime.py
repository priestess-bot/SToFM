"""Scoped FlagGems dispatch for reproducible SToFM inference.

The scope deliberately uses ``flag_gems.use_gems()`` rather than a global
registration.  It can therefore measure FlagOS lifecycle cost separately from
steady-state inference and it does not leave ATen overrides installed after an
extraction job returns.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Iterator, Optional, Tuple


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


@contextmanager
def flagos_inference_scope(
    mode: str,
    *,
    enabled: bool = True,
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
            reason="FlagOS dispatch is inference-only; training or autograd is active",
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

    try:
        import flag_gems
    except ImportError as exc:
        raise RuntimeError(
            f"flagos_mode='{normalized}' requires the pinned FlagGems package"
        ) from exc

    # FlagGems' registrar owns a process-global ATen Library. Serializing
    # outer scopes prevents two Python threads from destroying each other's
    # temporary registration while retaining re-entrant use within one scope.
    with _FLAGGEMS_SCOPE_LOCK:
        with flag_gems.use_gems(include=STOFM_ATEN_ALLOWLIST):
            record = FlagOSRuntimeDispatch(
                mode=normalized,
                active=True,
                registered_aten_ops=_registered_aten_ops(flag_gems),
                reason="scoped FlagGems ATen dispatch",
            )
            token = _ACTIVE_SCOPE.set(record)
            try:
                yield record
            finally:
                _ACTIVE_SCOPE.reset(token)
