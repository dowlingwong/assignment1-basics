import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    The normalization computation is done in float32 for numerical stability,
    then cast back to the input dtype before multiplying by the learned weight.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for the normalization step (avoids fp16/bf16 overflow).
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = (x_fp32 / rms).to(x.dtype)
        return x_normed * self.weight
