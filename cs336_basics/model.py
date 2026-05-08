"""Transformer language model components.

State dict key layout (must match the reference checkpoint):

TransformerLM:
  token_embeddings.weight            (vocab_size, d_model)
  layers.{i}.ln1.weight              (d_model,)
  layers.{i}.attn.q_proj.weight      (d_model, d_model)
  layers.{i}.attn.k_proj.weight      (d_model, d_model)
  layers.{i}.attn.v_proj.weight      (d_model, d_model)
  layers.{i}.attn.output_proj.weight (d_model, d_model)
  layers.{i}.ln2.weight              (d_model,)
  layers.{i}.ffn.w1.weight           (d_ff, d_model)
  layers.{i}.ffn.w2.weight           (d_model, d_ff)
  layers.{i}.ffn.w3.weight           (d_ff, d_model)
  ln_final.weight                    (d_model,)
  lm_head.weight                     (vocab_size, d_model)
"""

import torch
import torch.nn as nn

from cs336_basics.modules.attention import MHA
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.ffn import SwiGLU
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with RoPE causal attention and SwiGLU FFN.

    Layout per block (pre-norm / GPT-NeoX style):
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MHA(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            token_positions: (batch, seq_len) or None
        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-norm attention residual
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        # Pre-norm FFN residual
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Full autoregressive Transformer language model.

    Processes token indices through an embedding, a stack of TransformerBlocks,
    a final RMSNorm, and a linear LM head to produce unnormalized logits.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # Weight shape (vocab_size, d_model) — same convention as the embedding table.
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_indices: (batch, seq_len) integer token ids, seq_len ≤ context_length
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = in_indices.shape

        # Absolute positions for RoPE (rows are identical: 0, 1, ..., seq_len-1)
        token_positions = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(in_indices)            # (batch, seq, d_model)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)                             # (batch, seq, d_model)
        logits = self.lm_head(x)                         # (batch, seq, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive token generation with temperature and top-p sampling.

        Args:
            prompt_ids: (1, prompt_len) seed token ids
            max_new_tokens: number of tokens to generate
            temperature: softmax temperature (1.0 = no change, <1 = sharper)
            top_p: nucleus sampling threshold (1.0 = disabled)
        Returns:
            (1, prompt_len + max_new_tokens) token ids
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate context to context_length
            ctx = ids[:, -self.context_length:]
            logits = self.forward(ctx)          # (1, seq, vocab)
            next_logits = logits[:, -1, :]      # (1, vocab)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            probs = torch.softmax(next_logits, dim=-1)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens once cumulative probability exceeds top_p
                remove = cumulative - sorted_probs > top_p
                sorted_probs[remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = torch.multinomial(probs, 1)  # (1, 1)

            ids = torch.cat([ids, next_token], dim=-1)

        return ids
