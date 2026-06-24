# Interactive Transformer Attention & Entropy Visualizer

> Explore how a causal language model distributes its attention across text — in the browser, with zero dependencies.

![Demo](docs/demo.gif)

## Why attention entropy matters

**Low entropy** means a token's attention is sharply focused on a few positions — the model "knows where to look." **High entropy** means attention is spread thin, indicating uncertainty or distributed context gathering. This tool makes both patterns immediately visible, helping you debug model focus and understand how information flows through the context window.

## Three visualization modes

| Mode | What it shows |
|---|---|
| **Entropy (per word)** | How uncertain each word's attention distribution is. Blue = below median (focused), red = above (diffuse). |
| **Attention received (per word)** | How much attention each word attracts from later tokens, normalized for causal reachability. |
| **Backward attention** | Click any word to draw arcs to its top-10 most-attended preceding words, with a ranked summary panel. |

## Tech stack

- **PyTorch** — model inference with fp16, eager attention
- **Hugging Face Transformers** — model & tokenizer loading
- **NumPy** — metric aggregation
- **Vanilla JS + SVG** — client-side visualization (no frameworks, no build step)

## Quick start

```bash
# Install dependencies
uv sync

# Run the pipeline (loads model, computes metrics, emits HTML)
uv run python main.py

# Open the result
open results/heatmap.html
```

### Custom input

```bash
# Use a different model
uv run python main.py --model gpt2

# Analyze your own text
uv run python main.py --text-file my_text.txt

# Change output location
uv run python main.py --output custom.html

# Adjust top-k for backward attention
uv run python main.py --top-k 15
```

## Use cases

- **LLM researchers** — Identify attention sinks, debug why certain tokens absorb or starve for attention, and study how architectural choices affect attention patterns.
- **NLP practitioners** — Audit model focus: does the model attend to task-relevant words, or get distracted by punctuation and formatting tokens?
- **Educators** — Demonstrate how causal (autoregressive) attention works in transformer models, with an intuitive, interactive interface.

## How it works

1. **Model inference** — The text is tokenized and passed through a causal LM with `output_attentions=True`. The last decoder layer's attention weights are head-averaged into a single `(seq_len, seq_len)` matrix.

2. **Token-to-word mapping** — Tokenizer offset mappings are projected onto whitespace-delimited word spans. A token's last character is used for matching, since leading-space tokens (e.g. `Ġthe`) start on the space, not the word.

3. **Reachability normalization** — Raw column-sum attention is dominated by position (early tokens are reachable by all later tokens). Dividing each column by the number of tokens that can attend to it removes this positional bias.

4. **Outlier handling** — Values are clamped to Tukey IQR fences `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, then mapped through a log-stretched diverging colormap centered on the median.

5. **Backward attention** — A word-level attention matrix is computed via `S @ A @ S^T` (where `S` is a token-to-word selector matrix). For each word, the top-k preceding words by attention weight are precomputed and embedded as JSON.

## Project structure

```
main.py          # CLI entry point — orchestrates the full pipeline
model.py         # Model & tokenizer loading (offline cache support, retry logic)
metrics.py       # Attention entropy, reachability normalization, backward attention
text_utils.py    # Word segmentation, token-to-word mapping, metric aggregation
visualize.py     # HTML template + builder (diverging colormap, IQR clamping, SVG arcs)
sample.txt       # Default input text
```

## License

MIT
