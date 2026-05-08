"""Training utilities for the CS336 language model assignment.

Provides:
  cross_entropy_loss  — numerically stable cross-entropy
  get_batch           — random LM batch sampler
  AdamW               — AdamW optimizer with decoupled weight decay
  get_lr_cosine_schedule — cosine LR schedule with linear warmup
  gradient_clipping   — combined L2 gradient clipping
  save_checkpoint     — serialize model + optimizer + iteration
  load_checkpoint     — restore model + optimizer, return iteration
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    inputs: torch.Tensor,   # (batch, vocab_size) unnormalized logits
    targets: torch.Tensor,  # (batch,) class indices
) -> torch.Tensor:
    """Numerically stable mean cross-entropy loss.

    Uses the log-sum-exp trick: subtract the per-row maximum before computing
    log-softmax so that we never overflow inside exp().
    """
    # Shift for numerical stability: subtract per-example max.
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))             # (batch,)
    log_probs = shifted - log_sum_exp.unsqueeze(-1)                     # (batch, vocab)

    # Gather the log-prob of the correct class and take the negative mean.
    nll = -log_probs[torch.arange(len(targets), device=inputs.device), targets]
    return nll.mean()


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random language-modeling batch from a token ID array.

    Args:
        dataset: 1-D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each input sequence.
        device: PyTorch device string.

    Returns:
        (x, y) both of shape (batch_size, context_length) as LongTensors.
        y[i] == x[i] shifted one token to the right (next-token prediction).
    """
    n = len(dataset)
    # Valid start indices: [0, n - context_length - 1] so that x AND y fit.
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=(batch_size,))

    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])

    x_t = torch.from_numpy(x.astype(np.int64)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    return x_t, y_t


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------

class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay.

    Implements the update from Loshchilov & Hutter (2019):
      m_t = β1 m_{t-1} + (1 - β1) g_t
      v_t = β2 v_{t-1} + (1 - β2) g_t²
      θ_t = θ_{t-1} - α [ m̂_t / (√v̂_t + ε)  +  λ θ_{t-1} ]

    where m̂_t, v̂_t are bias-corrected first and second moments, and the
    weight-decay term λ acts directly on the parameter (decoupled from Adam).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Moment updates
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t
                step_size = lr / bias_corr1

                # Decoupled weight decay (applied to raw parameter, not to m/v)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Parameter update
                denom = (v.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing schedule with linear warmup.

    Three phases:
      1. Linear warmup:  it < warmup_iters
         lr = max_lr * it / warmup_iters
      2. Cosine decay:   warmup_iters ≤ it < cosine_cycle_iters
         lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
      3. Flat minimum:   it ≥ cosine_cycle_iters
         lr = min_lr
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    # Cosine decay in [warmup_iters, cosine_cycle_iters)
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip gradients so their combined L2 norm is at most max_l2_norm.

    Only considers parameters that actually have a gradient.
    Modifies gradients in-place.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    total_norm = torch.sqrt(
        sum(p.grad.detach().norm() ** 2 for p in params_with_grad)
    )

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model weights, optimizer state, and iteration number."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state from a checkpoint file.

    Returns:
        The iteration number that was saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
