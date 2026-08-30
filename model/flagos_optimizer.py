"""FlagOS optimizer adapters used by SToFM training experiments."""

from __future__ import annotations

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
        if dispatch is None or "_fused_adamw_" not in dispatch.registered_aten_ops:
            raise RuntimeError(
                "FlagOSFusedAdamW.step() requires an active FlagOS training scope "
                "with '_fused_adamw_' registered"
            )

        with torch.profiler.record_function("Optimizer.step#FlagOSFusedAdamW.step"):
            for group in self.param_groups:
                params = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []
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
                            (), dtype=torch.float32, device=parameter.device
                        )
                        state["exp_avg"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    state["step"].add_(1.0)
                    params.append(parameter)
                    grads.append(gradient)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])

                if not params:
                    continue
                beta1, beta2 = group["betas"]
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
