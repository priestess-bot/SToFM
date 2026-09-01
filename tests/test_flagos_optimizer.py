import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from model.flagos_optimizer import FlagOSFusedAdamW
from model.flagos_runtime import flagos_training_scope


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlagOS fused AdamW requires CUDA")
def test_flagos_fused_adamw_matches_torch_fused_state_and_parameters():
    torch.manual_seed(67)
    reference_parameters = [
        torch.nn.Parameter(torch.randn(shape, device="cuda"))
        for shape in ((257,), (17, 19), (3,))
    ]
    actual_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in reference_parameters
    ]
    reference = torch.optim.AdamW(
        reference_parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    actual = FlagOSFusedAdamW(
        actual_parameters, lr=1e-3, weight_decay=0.01
    )

    with flagos_training_scope(
        strict=True, include=("_fused_adamw_", "add_")
    ):
        for _ in range(3):
            gradients = [torch.randn_like(parameter) for parameter in reference_parameters]
            for reference_parameter, actual_parameter, gradient in zip(
                reference_parameters, actual_parameters, gradients
            ):
                reference_parameter.grad = gradient.clone()
                actual_parameter.grad = gradient.clone()
            reference.step()
            actual.step()
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    for reference_parameter, actual_parameter in zip(
        reference_parameters, actual_parameters
    ):
        torch.testing.assert_close(actual_parameter, reference_parameter, rtol=0.0, atol=0.0)
        reference_state = reference.state[reference_parameter]
        actual_state = actual.state[actual_parameter]
        torch.testing.assert_close(
            actual_state["exp_avg"], reference_state["exp_avg"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            actual_state["exp_avg_sq"],
            reference_state["exp_avg_sq"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_state["step"], reference_state["step"], rtol=0.0, atol=0.0
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("FLAGGEMS_STOFM_VENDOR_LIBRARY"),
    reason="built Vendor library is required",
)
def test_vendor_adamw_splits_parameter_lists_larger_than_kernel_pack():
    torch.manual_seed(71)
    reference_parameters = [
        torch.nn.Parameter(torch.randn(3, device="cuda")) for _ in range(70)
    ]
    actual_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in reference_parameters
    ]
    reference = torch.optim.AdamW(
        reference_parameters, lr=1e-3, weight_decay=0.01, fused=True
    )
    actual = FlagOSFusedAdamW(
        actual_parameters, lr=1e-3, weight_decay=0.01
    )
    with flagos_training_scope(strict=True, include=(), gemm_backend="vendor"):
        for _ in range(2):
            gradients = [torch.randn_like(parameter) for parameter in reference_parameters]
            for reference_parameter, actual_parameter, gradient in zip(
                reference_parameters, actual_parameters, gradients
            ):
                reference_parameter.grad = gradient.clone()
                actual_parameter.grad = gradient.clone()
            reference.step()
            actual.step()
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    for reference_parameter, actual_parameter in zip(
        reference_parameters, actual_parameters
    ):
        torch.testing.assert_close(
            actual_parameter, reference_parameter, rtol=0.0, atol=2e-6
        )
