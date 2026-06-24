"""Attention and entropy metric extraction from transformer outputs."""

from typing import Any

import numpy as np
import torch


def compute_token_metrics(
    outputs: Any,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Compute per-token row entropy and reachability-normalised attention.

    Row entropy measures how uncertain the model's attention distribution is
    *from* each token.  Received attention is the column sum *to* each token,
    divided by how many tokens can causally attend to it (removing positional
    bias so the heatmap reflects genuine focus, not sequence position).

    Args:
        outputs: Raw model outputs with ``attentions`` populated.

    Returns:
        A tuple ``(entropy, received, attn_matrix)`` where *entropy* and
        *received* are 1-D ``np.ndarray`` of length ``seq_len`` and
        *attn_matrix* is the head-averaged ``torch.Tensor`` of shape
        ``(seq_len, seq_len)``.
    """
    attn = outputs.attentions[-1][0].mean(dim=0).float()
    attn = torch.nan_to_num(attn)
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)
    entropy = torch.nan_to_num(entropy)
    seq_len = attn.size(-1)
    positions = torch.arange(seq_len, device=attn.device)
    reachability = (seq_len - positions).clamp(min=1)
    received = attn.sum(dim=0) / reachability
    received = torch.nan_to_num(received)
    return entropy.cpu().numpy(), received.cpu().numpy(), attn


def build_backward_attention(
    attn_matrix: torch.Tensor,
    word_units: list[list],
    top_k: int = 10,
) -> list[list[list[float]]]:
    """Compute word-level backward attention: top-*k* preceding words per word.

    Builds a word-by-word attention matrix ``W`` via ``S @ A @ Sᵀ`` where *S*
    is a selector matrix mapping tokens to words.  Each row is normalised,
    then the top-*k* preceding entries (causal mask: *j < i*) are extracted.

    Args:
        attn_matrix: Head-averaged token-level attention ``(seq_len, seq_len)``.
        word_units: Output of :func:`text_utils.assign_tokens_to_words`.
        top_k: Number of preceding words to retain per target word.

    Returns:
        ``backward[i] = [[j, weight], ...]`` for each word *i*, sorted
        descending by weight, containing only words *j < i*.
    """
    n_tok = attn_matrix.size(-1)
    n_words = len(word_units)
    device = attn_matrix.device
    S = torch.zeros((n_words, n_tok), device=device)
    for wi, unit in enumerate(word_units):
        for ti in unit[4]:
            S[wi, ti] = 1.0
    W = S @ attn_matrix @ S.t()
    W = torch.nan_to_num(W)
    row_sum = W.sum(dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    W = W / row_sum
    W = W.cpu().numpy()

    backward: list[list[list[float]]] = []
    for i in range(n_words):
        row = W[i, :i]
        if row.size == 0:
            backward.append([])
            continue
        k = min(top_k, row.size)
        top = np.argpartition(row, -k)[-k:]
        top = top[np.argsort(row[top])[::-1]]
        backward.append([[int(j), float(row[j])] for j in top if row[j] > 0])
    return backward
