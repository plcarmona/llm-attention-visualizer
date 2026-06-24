"""Entry point — run the full attention-visualizer pipeline.

Usage::

    uv run python main.py                      # defaults
    uv run python main.py --model gpt2         # different model
    uv run python main.py --text-file my.txt   # custom text
    uv run python main.py --output out.html    # custom output path
"""

import argparse
from pathlib import Path

import torch

from metrics import build_backward_attention, compute_token_metrics
from model import load_model
from text_utils import aggregate_word_metrics, assign_tokens_to_words, split_into_word_units
from visualize import build_html

DEFAULT_MODEL = "slicexai/Llama3.1-elm-turbo-3B-instruct"
SAMPLE_TEXT = Path(__file__).parent / "sample.txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive attention & entropy heatmap generator for causal LLMs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=SAMPLE_TEXT,
        help="Path to input text file (default: sample.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/heatmap.html"),
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k preceding words in backward-attention mode (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline: load model, compute metrics, emit HTML."""
    args = parse_args()
    torch.set_grad_enabled(False)

    text = args.text_file.read_text(encoding="utf-8")
    tokenizer, model = load_model(args.model)

    enc = tokenizer(text, return_offsets_mapping=True)
    # Prepend BOS so the attention sink lands on it (offset (0,0), excluded
    # from words) instead of on the first real word.
    bos = tokenizer.bos_token_id
    ids = ([bos] + enc["input_ids"]) if bos is not None else list(enc["input_ids"])
    offsets = (
        [(0, 0)] + enc["offset_mapping"]
        if bos is not None
        else enc["offset_mapping"]
    )
    input_ids = torch.tensor([ids], device=model.device)
    outputs = model(input_ids=input_ids)

    entropy_tok, attn_recv, attn_matrix = compute_token_metrics(outputs)
    word_units = assign_tokens_to_words(offsets, split_into_word_units(text))
    word_entropy, word_attn = aggregate_word_metrics(word_units, entropy_tok, attn_recv)
    backward = build_backward_attention(attn_matrix, word_units, top_k=args.top_k)

    print(f"Words: {len(word_units)}  Tokens: {len(offsets)}")
    print(f"Entropy range: {min(word_entropy):.4f} .. {max(word_entropy):.4f}")
    print(f"Attention range: {min(word_attn):.6f} .. {max(word_attn):.6f}")
    bw_total = sum(len(b) for b in backward)
    print(f"Backward links: {bw_total} (avg {bw_total / max(len(backward), 1):.1f}/word)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_html(word_units, word_entropy, word_attn, backward),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
