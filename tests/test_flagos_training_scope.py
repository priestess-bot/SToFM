import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from model.flagos_runtime import (
    STOFM_TRAINING_ALLOWLIST,
    current_flagos_runtime_dispatch,
    current_flagos_training_dispatch,
    flagos_training_scope,
)


def test_torch_training_scope_is_explicitly_inactive_and_cleans_context():
    assert current_flagos_runtime_dispatch() is None
    with flagos_training_scope("torch", strict=True) as dispatch:
        assert dispatch.phase == "training"
        assert not dispatch.active
        assert dispatch.strict
        assert current_flagos_training_dispatch() is None
    assert current_flagos_runtime_dispatch() is None
    assert current_flagos_training_dispatch() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagGems training scope requires CUDA")
def test_v100_training_scope_registers_curated_ops_and_cleans_context():
    with flagos_training_scope(strict=True) as dispatch:
        assert dispatch.active
        assert dispatch.phase == "training"
        assert dispatch.strict
        assert {"mm", "addmm", "bmm", "layer_norm", "vector_norm"}.issubset(
            set(STOFM_TRAINING_ALLOWLIST)
        )
        assert set(STOFM_TRAINING_ALLOWLIST).issubset(
            set(dispatch.registered_aten_ops)
        )
        assert "layer_norm" in dispatch.registered_aten_ops
        assert "vector_norm" in dispatch.registered_aten_ops
        assert current_flagos_training_dispatch() == dispatch
    assert current_flagos_runtime_dispatch() is None
