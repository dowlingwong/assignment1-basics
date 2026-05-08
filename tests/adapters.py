from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    """
    from cs336_basics.modules.linear import Linear

    linear = Linear(d_in, d_out)
    linear.weight.data = weights
    linear.eval()
    with torch.no_grad():
        return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    from cs336_basics.modules.embedding import Embedding

    emb = Embedding(vocab_size, d_model)
    emb.weight.data = weights
    emb.eval()
    with torch.no_grad():
        return emb(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return the output."""
    from cs336_basics.modules.ffn import SwiGLU

    ffn = SwiGLU(d_model, d_ff)
    ffn.w1.weight.data = w1_weight
    ffn.w2.weight.data = w2_weight
    ffn.w3.weight.data = w3_weight
    ffn.eval()
    with torch.no_grad():
        return ffn(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return the output of
    scaled dot product attention.
    """
    from cs336_basics.modules.attention import scaled_dot_product_attention

    with torch.no_grad():
        return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """Multi-head self-attention without RoPE."""
    from cs336_basics.modules.attention import MHA

    attn = MHA(d_model, num_heads, use_rope=False)
    attn.q_proj.weight.data = q_proj_weight
    attn.k_proj.weight.data = k_proj_weight
    attn.v_proj.weight.data = v_proj_weight
    attn.output_proj.weight.data = o_proj_weight
    attn.eval()
    with torch.no_grad():
        return attn(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """Multi-head self-attention with RoPE."""
    from cs336_basics.modules.attention import MHA

    attn = MHA(d_model, num_heads, use_rope=True, theta=theta, max_seq_len=max_seq_len)
    attn.q_proj.weight.data = q_proj_weight
    attn.k_proj.weight.data = k_proj_weight
    attn.v_proj.weight.data = v_proj_weight
    attn.output_proj.weight.data = o_proj_weight
    attn.eval()
    with torch.no_grad():
        return attn(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """Run RoPE for a given input tensor."""
    from cs336_basics.modules.rope import RoPEEmbedding

    rope = RoPEEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    with torch.no_grad():
        return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """Run a single pre-norm Transformer block."""
    from cs336_basics.model import TransformerBlock

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    block.load_state_dict(weights)
    block.eval()
    with torch.no_grad():
        return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Run a full Transformer language model forward pass."""
    from cs336_basics.model import TransformerLM

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Run RMSNorm with the given weights."""
    from cs336_basics.modules.norm import RMSNorm

    norm = RMSNorm(d_model, eps=eps)
    norm.weight.data = weights
    norm.eval()
    with torch.no_grad():
        return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Apply SiLU element-wise."""
    from cs336_basics.modules.ffn import silu

    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random language modeling batch."""
    from cs336_basics.training import get_batch

    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """Numerically stable softmax."""
    from cs336_basics.modules.attention import stable_softmax

    return stable_softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Numerically stable mean cross-entropy loss."""
    from cs336_basics.training import cross_entropy_loss

    return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip combined L2 norm of all parameter gradients to max_l2_norm."""
    from cs336_basics.training import gradient_clipping

    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """Return the AdamW optimizer class."""
    from cs336_basics.training import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Return the learning rate at iteration `it` under a cosine schedule."""
    from cs336_basics.training import get_lr_cosine_schedule

    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Save model, optimizer, and iteration to a checkpoint file."""
    from cs336_basics.training import save_checkpoint

    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load a checkpoint and return the saved iteration number."""
    from cs336_basics.training import load_checkpoint

    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Return a BPETokenizer initialized with the given vocab and merges."""
    from cs336_basics.tokenizer.tokenizer import BPETokenizer

    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer and return (vocab, merges)."""
    from cs336_basics.tokenizer.tokenizer import train_bpe

    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
