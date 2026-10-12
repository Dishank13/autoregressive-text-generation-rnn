"""Train the four models under identical conditions and evaluate them on the test split.

Run from the repository root:  python -m src.train
"""
import csv
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import describe, prepare
from .models import NextWordModel, count_parameters


@torch.no_grad()
def evaluate(model, loader):
    """Mean cross-entropy, perplexity, top-1 and top-5 next-word accuracy."""
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        top = logits.topk(5, dim=1).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.size(0)
    loss = loss_sum / n
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": top1 / n, "top5_accuracy": top5 / n}


def train_model(name, vocab_size, loaders):
    torch.manual_seed(config.SEED)
    model = NextWordModel(name, vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "val_perplexity": [], "val_accuracy": [], "epoch_time": []}
    best_val, best_epoch = float("inf"), 0
    checkpoint = config.CHECKPOINT_DIR / f"{name}.pt"

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        start = time.perf_counter()
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        epoch_time = time.perf_counter() - start

        val = evaluate(model, loaders["val"])
        history["train_loss"].append(loss_sum / n)
        history["val_loss"].append(val["loss"])
        history["val_perplexity"].append(val["perplexity"])
        history["val_accuracy"].append(val["accuracy"])
        history["epoch_time"].append(epoch_time)
        if val["loss"] < best_val:
            best_val, best_epoch = val["loss"], epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"{name:9s} epoch {epoch:2d}  train loss {loss_sum / n:.3f}  val loss {val['loss']:.3f}  "
              f"val ppl {val['perplexity']:.1f}  val acc {val['accuracy']:.3f}  {epoch_time:.1f}s", flush=True)

    # The checkpoint with the lowest validation loss is the one evaluated and used for generation.
    model.load_state_dict(torch.load(checkpoint))
    test = evaluate(model, loaders["test"])
    total_time = sum(history["epoch_time"])
    summary = {
        "model": name,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "train_loss_at_best": history["train_loss"][best_epoch - 1],
        "val_loss": best_val,
        "val_perplexity": math.exp(best_val),
        "test_loss": test["loss"],
        "test_perplexity": test["perplexity"],
        "test_accuracy": test["accuracy"],
        "test_top5_accuracy": test["top5_accuracy"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "train_time_total_s": total_time,
        "train_time_per_epoch_s": total_time / config.EPOCHS,
        "train_time_to_best_s": sum(history["epoch_time"][:best_epoch]),
    }
    return history, summary


def plot_curves(histories):
    epochs = range(1, config.EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for name, h in histories.items():
        axes[0].plot(epochs, h["train_loss"], label=name)
        axes[1].plot(epochs, h["val_loss"], label=name)
        axes[2].plot(epochs, h["val_accuracy"], label=name)
    for ax, title, ylabel in zip(axes, ["Training loss", "Validation loss", "Validation accuracy"],
                                 ["cross-entropy", "cross-entropy", "top-1 accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_comparison(summaries):
    names = [s["model"] for s in summaries]
    panels = [
        ("test_perplexity", "Test perplexity (lower is better)"),
        ("test_accuracy", "Test top-1 accuracy (higher is better)"),
        ("train_time_total_s", f"Training time for {config.EPOCHS} epochs (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(names)]
    for ax, (key, title) in zip(axes, panels):
        values = [s[key] for s in summaries]
        bars = ax.bar(names, values, color=colors)
        ax.bar_label(bars, fmt="%.3f" if key == "test_accuracy" else "%.1f")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare()
    vocab = data["vocab"]

    stats = describe(data)
    (config.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    loaders = {}
    for split, (x, y) in data["sequences"].items():
        generator = torch.Generator().manual_seed(config.SEED)  # same batch order for every model
        loaders[split] = DataLoader(TensorDataset(x, y), batch_size=config.BATCH_SIZE,
                                    shuffle=(split == "train"), generator=generator)

    histories, summaries = {}, []
    for name in config.MODEL_NAMES:
        loaders["train"].generator.manual_seed(config.SEED)
        histories[name], summary = train_model(name, len(vocab), loaders)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    (config.RESULTS_DIR / "training_history.json").write_text(json.dumps(histories, indent=2))
    with open(config.RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_curves(histories)
    plot_comparison(summaries)


if __name__ == "__main__":
    main()
