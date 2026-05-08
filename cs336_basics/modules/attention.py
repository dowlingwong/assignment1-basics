import torch
import torch.nn as nn


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax: subtract max before exp to avoid overflow."""
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    return exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Args:
        query: (..., queries, d_k)
        key:   (..., keys,    d_k)
        value: (..., keys,    d_v)
        mask:  (..., queries, keys) boolean — True = attend, False = mask out
    Returns:
        (..., queries, d_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        # mask=False means "do not attend" → fill with -inf so softmax → 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = stable_softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)


class MHA(nn.Module):
    """Causal multi-head self-attention with optional RoPE.

    Projection weight names match the reference state dict:
      q_proj, k_proj, v_proj, output_proj
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        from cs336_basics.modules.linear import Linear
        from cs336_basics.modules.rope import RoPEEmbedding

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Names must match the reference state dict keys.
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular boolean mask: True = attend."""
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            token_positions: (batch, seq_len) or (1, seq_len) integer positions for RoPE.
                             If None, positions 0..seq_len-1 are used.
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        causal_mask = self._causal_mask(seq_len, x.device)  # (1, 1, seq, seq)

        # Project and split into heads: (batch, heads, seq, d_k)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            # Expand token_positions to include the heads dimension so RoPE
            # can broadcast correctly: (batch, seq) → (batch, 1, seq)
            rope_positions: torch.Tensor | None
            if token_positions is not None:
                rope_positions = token_positions.unsqueeze(-2)  # (..., 1, seq)
            else:
                rope_positions = None  # RoPE will create (seq,) and broadcast
            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)

        # Attention + output projection
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        # Merge heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.output_proj(attn_out)
