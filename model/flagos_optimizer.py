"""FlagOS optimizer adapters used by SToFM training experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import torch


class FlagOSFusedAdamW(torch.optim.Optimizer):
    """Dense FP32 AdamW backed by FlagGems' ``aten::_fused_adamw_`` kernel.

    The FlagGems implementation fuses all elementwise work for one parameter
    into one Triton launch. It still launches once per parameter and therefore
    must not be described as a cross-parameter foreach kernel.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                betas=tuple(float(value) for value in betas),
                eps=float(eps),
                weight_decay=float(weight_decay),
                maximize=bool(maximize),
            ),
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        from .flagos_runtime import current_flagos_training_dispatch

        dispatch = current_flagos_training_dispatch()
        if dispatch is None:
            raise RuntimeError(
                "FlagOSFusedAdamW.step() requires an active FlagOS training scope "
                "with a registered FlagOS optimizer implementation"
            )
        vendor_backend = getattr(dispatch, "gemm_backend", "triton") == "vendor"
        if not vendor_backend and "_fused_adamw_" not in dispatch.registered_aten_ops:
            raise RuntimeError(
                "FlagOSFusedAdamW.step() requires '_fused_adamw_' in the active "
                "FlagOS training scope"
            )

        with torch.profiler.record_function("Optimizer.step#FlagOSFusedAdamW.step"):
            for group in self.param_groups:
                params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []
                host_steps = []
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_sparse:
                        raise RuntimeError("FlagOSFusedAdamW does not support sparse gradients")
                    if parameter.device.type != "cuda" or parameter.dtype != torch.float32:
                        raise RuntimeError("FlagOSFusedAdamW requires dense FP32 CUDA parameters")
                    if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
                        raise RuntimeError("gradient device and dtype must match its parameter")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = torch.zeros(
                            (),
                            dtype=torch.float32,
                            device=parameter.device,
                        )
                        state["exp_avg"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    if vendor_backend:
                        # Keep a host mirror so grouped tensors with missing
                        # gradients preserve AdamW's per-parameter step without
                        # launching one CUDA scalar update per parameter.
                        if "_flagos_host_step" not in state:
                            state["_flagos_host_step"] = int(state["step"].item())
                        state["_flagos_host_step"] += 1
                        host_step = int(state["_flagos_host_step"])
                    else:
                        state["step"].add_(1.0)
                        host_step = None
                    params.append(parameter)
                    grads.append(gradient)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])
                    host_steps.append(host_step)

                if not params:
                    continue
                beta1, beta2 = group["betas"]
                if vendor_backend:
                    from flag_gems.experimental_ops.vendor_gemm import vendor_adamw_multi

                    # Usually all SToFM parameters have gradients and this is
                    # one launch.  Grouping keeps the adapter correct for a
                    # partially frozen model as well.
                    grouped = defaultdict(list)
                    for index, host_step in enumerate(host_steps):
                        grouped[host_step].append(index)
                    for host_step, indices in grouped.items():
                        for start in range(0, len(indices), 64):
                            chunk = indices[start : start + 64]
                            vendor_adamw_multi(
                                [params[index] for index in chunk],
                                [grads[index] for index in chunk],
                                [exp_avgs[index] for index in chunk],
                                [exp_avg_sqs[index] for index in chunk],
                                [state_steps[index] for index in chunk],
                                lr=group["lr"],
                                beta1=beta1,
                                beta2=beta2,
                                weight_decay=group["weight_decay"],
                                eps=group["eps"],
                                step=float(host_step),
                                maximize=group["maximize"],
                            )
                else:
                    torch.ops.aten._fused_adamw_(
                        params,
                        grads,
                        exp_avgs,
                        exp_avg_sqs,
                        [],
                        state_steps,
                        lr=group["lr"],
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        amsgrad=False,
                        maximize=group["maximize"],
                    )
        return loss
