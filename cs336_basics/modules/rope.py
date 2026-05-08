import einops
import torch
import torch.nn as nn


class RoPEEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Applies a rotation to each (x_{2i}, x_{2i+1}) pair in the feature dimension
    based on the token position and a fixed frequency schedule.

    Works with tensors of any shape (..., seq_len, d_k):
      - If token_positions is None, creates [0, 1, ..., seq_len-1] as a 1-D tensor
        so it broadcasts over all leading dimensions automatically.
      - If token_positions is provided it must have shape (..., seq_len) matching
        the leading dims of the input tensor.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta_base = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq[i] = 1 / theta^(2i / d_k),  shape: (d_k // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent pairs: (x_1, x_2, ..., x_{d-1}, x_d)
                                → (-x_2, x_1, ..., -x_d, x_{d-1})"""
        x = einops.rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        return einops.rearrange(torch.stack((-x2, x1), dim=-1), "... d j -> ... (d j)")

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (..., seq_len, d_k) — query or key tensor
            token_positions: (..., seq_len) integer positions, or None to use 0..seq_len-1
        Returns:
            RoPE-encoded tensor of the same shape as x
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            # Shape (seq_len,) broadcasts against any number of leading dims.
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        # Cast to float for the frequency multiplication.
        # angles: (..., seq_len, d_k // 2)
        angles = torch.einsum("...i, j -> ...ij", token_positions.float(), self.inv_freq)

        # Repeat each frequency for both elements of the pair → (..., seq_len, d_k)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        return (x * cos) + (self._rotate_half(x) * sin)
