import torch
import torch.nn as nn


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network, used in LLaMA-style transformers.

    Formula:  output = w2( SiLU(w1(x)) ⊙ w3(x) )

    Weight naming matches the reference state dict:
      w1: (d_ff, d_model)  — "gate" path (inside SiLU)
      w2: (d_model, d_ff)  — down-projection
      w3: (d_ff, d_model)  — "up" path (multiplied by SiLU output)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        from cs336_basics.modules.linear import Linear

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
