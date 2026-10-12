"""Generate text from the trained models at several sampling temperatures and measure it.

Run from the repository root (after training):  python -m src.generate
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import config
from .data import detokenize, prepare
from .models import load_model


@torch.no_grad()
def generate(model, seed_ids, length, temperature, unk_idx, rng):
    """Autoregressive sampling: predict a word, append it, slide the context window, repeat."""
    ids = list(seed_ids)
    for _ in range(length):
        context = torch.tensor([ids[-config.SEQ_LEN:]])
        logits = model(context)[0] / temperature
        logits[unk_idx] = float("-inf")  # never emit the unknown-word token
        next_id = torch.multinomial(torch.softmax(logits, dim=0), 1, generator=rng).item()
        ids.append(next_id)
    return ids[len(seed_ids):]


def ngrams(tokens, n):
    return list(zip(*(tokens[i:] for i in range(n))))


def text_metrics(tokens, train_bigrams):
    """Automatic proxies for the qualitative properties discussed in the README.

    distinct-1 / distinct-2: unique unigrams / bigrams divided by their total (diversity).
    repetition: share of trigrams that already appeared earlier in the same sample.
    seen_bigrams: share of generated bigrams that occur in the training text (local coherence).
    """
    bigrams, trigrams = ngrams(tokens, 2), ngrams(tokens, 3)
    return {
        "distinct_1": len(set(tokens)) / len(tokens),
        "distinct_2": len(set(bigrams)) / len(bigrams),
        "repetition": 1 - len(set(trigrams)) / len(trigrams),
        "seen_bigrams": sum(b in train_bigrams for b in bigrams) / len(bigrams),
    }


def pick_seeds(test_ids, vocab):
    """Seed sequences: evenly spaced sentence openings from the unseen test split, without <unk>."""
    sentence_end = {vocab.stoi[p] for p in ".!?"}
    starts = [i + 1 for i, t in enumerate(test_ids) if t in sentence_end]
    candidates = [s for s in starts
                  if s + config.SEQ_LEN <= len(test_ids)
                  and vocab.unk_idx not in test_ids[s:s + config.SEQ_LEN]
                  and vocab.itos[test_ids[s]][0].isalpha()]
    step = len(candidates) // config.NUM_SEEDS
    return [test_ids[s:s + config.SEQ_LEN] for s in candidates[::step][:config.NUM_SEEDS]]


def plot_temperature(rows):
    panels = [("distinct_2", "Diversity (distinct-2)"), ("repetition", "Repetition (repeated trigrams)"),
              ("seen_bigrams", "Local coherence (bigrams seen in training)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (key, title) in zip(axes, panels):
        for name in config.MODEL_NAMES:
            values = [r[key] for r in rows if r["model"] == name]
            ax.plot(config.TEMPERATURES, values, marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("sampling temperature")
        ax.set_xticks(config.TEMPERATURES)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "temperature_analysis.png", dpi=150)
    plt.close(fig)


def main():
    data = prepare()
    vocab = data["vocab"]
    train_bigrams = set(ngrams(data["ids"]["train"], 2))
    seeds = pick_seeds(data["ids"]["test"], vocab)
    models = {name: load_model(name, len(vocab)) for name in config.MODEL_NAMES}

    rows = []
    samples = defaultdict(dict)  # (seed index, temperature) -> model -> first generated sample
    for name, model in models.items():
        for temperature in config.TEMPERATURES:
            totals = defaultdict(float)
            for seed_idx, seed in enumerate(seeds):
                for sample_idx in range(config.SAMPLES_PER_SETTING):
                    # the same random stream is used for every model, so only the model differs
                    rng = torch.Generator().manual_seed(config.SEED + 100 * seed_idx + sample_idx)
                    out = generate(model, seed, config.GEN_LEN, temperature, vocab.unk_idx, rng)
                    for key, value in text_metrics(out, train_bigrams).items():
                        totals[key] += value
                    if sample_idx == 0:
                        samples[(seed_idx, temperature)][name] = detokenize(vocab.decode(out))
            count = len(seeds) * config.SAMPLES_PER_SETTING
            row = {"model": name, "temperature": temperature}
            row.update({key: round(value / count, 4) for key, value in totals.items()})
            rows.append(row)
            print(row, flush=True)

    with open(config.RESULTS_DIR / "temperature_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_temperature(rows)

    lines = ["# Generated text samples", "",
             f"{config.GEN_LEN} words generated per sample. Seeds are taken from the held-out test split. "
             "For a given seed and temperature every model is sampled with the same random stream.", ""]
    for seed_idx, seed in enumerate(seeds):
        lines += [f"## Seed {seed_idx + 1}", "", f"> {detokenize(vocab.decode(seed))}", ""]
        for temperature in config.TEMPERATURES:
            lines += [f"### Temperature {temperature}", ""]
            for name in config.MODEL_NAMES:
                lines += [f"**{name}:** {samples[(seed_idx, temperature)][name]}", ""]
    (config.RESULTS_DIR / "generated_samples.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
