import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Token embedding table.

    Supports arbitrary leading dimensions on the input tensor, not just (batch, seq).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...) integer indices  →  (..., embedding_dim)
        original_shape = x.shape
        flat = x.reshape(-1)                                 # (total_tokens,)
        out = self.weight.index_select(0, flat)              # (total_tokens, embedding_dim)
        return out.reshape(*original_shape, self.embedding_dim)

    def _init_weight(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
