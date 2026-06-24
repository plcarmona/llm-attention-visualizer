"""Model loading utilities with local-cache support and retry logic."""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

T = TypeVar("T")


def _with_retry(fn: Callable[[], T], attempts: int = 15, delay: float = 3.0) -> T:
    """Call *fn* with retries on any exception.

    Args:
        fn: Zero-argument callable to execute.
        attempts: Maximum number of attempts before re-raising.
        delay: Seconds to sleep between attempts.

    Returns:
        The return value of *fn* on success.

    Raises:
        Exception: The last exception encountered after all attempts.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            print(f"[retry {i + 1}/{attempts}] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(delay)
    assert last is not None
    raise last


def _resolve_local_snapshot(model_id: str) -> str | None:
    """Return the on-disk snapshot directory for a cached model, or ``None``.

    Args:
        model_id: Hugging Face model identifier (e.g. ``"slicexai/..."``).

    Returns:
        Path to the snapshot directory if it exists locally, otherwise ``None``.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo = cache / ("models--" + model_id.replace("/", "--"))
    ref = repo / "refs" / "main"
    try:
        commit = ref.read_text(encoding="utf-8").strip()
        snap = repo / "snapshots" / commit
        if (snap / "config.json").is_file():
            return str(snap)
    except OSError:
        pass
    return None


def load_model(
    model_id: str,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a Hugging Face tokenizer and model in fp16 with eager attention.

    Prefers a local cache snapshot (offline) and falls back to a network
    download with retry logic.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        A ``(tokenizer, model)`` tuple.
    """
    token = os.environ.get("HF_TOKEN")
    kw: dict[str, str] = {"token": token} if token else {}
    snapshot = _resolve_local_snapshot(model_id)
    source = snapshot or model_id
    cache_only = {"local_files_only": True} if snapshot else {}
    print(f"Loading from: {source}")
    tokenizer = _with_retry(
        lambda: AutoTokenizer.from_pretrained(source, **kw, **cache_only)
    )
    model = _with_retry(
        lambda: AutoModelForCausalLM.from_pretrained(
            source,
            device_map="auto",
            torch_dtype=torch.float16,
            output_attentions=True,
            attn_implementation="eager",
            **kw,
            **cache_only,
        )
    )
    return tokenizer, model
