import torch
import torch.nn as nn


class Linear(nn.Module):
    """Linear layer without bias by default.

    Stores weight in (out_features, in_features) shape — the PyTorch convention —
    so that state dicts are directly compatible with reference checkpoints.
    Forward pass computes  x @ weight.T.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Shape: (out_features, in_features) — matches PyTorch / reference state dicts.
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) → (..., out_features)
        o = x @ self.weight.T
        if self.bias is not None:
            o = o + self.bias
        return o

    def _init_weight(self) -> None:
        std = 1.0 / (2 * (self.in_features + self.out_features) ** 0.5)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
