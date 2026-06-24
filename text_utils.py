"""Text segmentation and token-to-word mapping utilities."""

import bisect
import re

import numpy as np

PARA_BREAK_NEWLINES = 1


def split_into_word_units(text: str) -> list[list]:
    """Split *text* into whitespace-delimited word units with char spans.

    Each unit is ``[paragraph_index, char_start, char_end, word_text, []]``.
    A paragraph break is detected when ``PARA_BREAK_NEWLINES`` consecutive
    newlines appear between two words.

    Args:
        text: Raw input text.

    Returns:
        List of word-unit lists (mutable; the last element is populated by
        :func:`assign_tokens_to_words`).
    """
    units: list[list] = []
    para = 0
    pos = 0
    for m in re.finditer(r"\S+", text):
        if "\n" * PARA_BREAK_NEWLINES in text[pos : m.start()] and units:
            para += 1
        units.append([para, m.start(), m.end(), m.group(), []])
        pos = m.end()
    return units


def assign_tokens_to_words(
    offsets: list[tuple[int, int]],
    word_units: list[list],
) -> list[list]:
    """Map tokenizer offset pairs onto the word units they belong to.

    A token's **last** character is used for matching because leading-space
    tokens (e.g. ``"Ġthe"``) start on a space character, not the word itself.

    Args:
        offsets: List of ``(start, end)`` char offsets from the tokenizer.
        word_units: Output of :func:`split_into_word_units` (mutated in place).

    Returns:
        The same *word_units* list (each unit's last element now contains
        token indices).
    """
    starts = [w[1] for w in word_units]
    ends = [w[2] for w in word_units]
    for ti, (ts, te) in enumerate(offsets):
        if ts == 0 and te == 0:
            continue
        p = te - 1
        idx = bisect.bisect_right(starts, p) - 1
        if 0 <= idx < len(word_units) and p < ends[idx]:
            word_units[idx][4].append(ti)
    return word_units


def aggregate_word_metrics(
    word_units: list[list],
    entropy_tok: np.ndarray,
    attn_recv: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Aggregate per-token metrics to the word level.

    Entropy is averaged across a word's tokens; received attention is summed.

    Args:
        word_units: Word units with token indices assigned.
        entropy_tok: Per-token entropy values.
        attn_recv: Per-token received-attention values.

    Returns:
        ``(word_entropy, word_attn)`` — two lists of floats, one per word.
    """
    word_entropy: list[float] = []
    word_attn: list[float] = []
    for unit in word_units:
        tidxs = unit[4]
        if tidxs:
            word_entropy.append(float(np.mean([entropy_tok[i] for i in tidxs])))
            word_attn.append(float(np.sum([attn_recv[i] for i in tidxs])))
        else:
            word_entropy.append(0.0)
            word_attn.append(0.0)
    return word_entropy, word_attn
