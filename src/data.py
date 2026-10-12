"""Loading, cleaning and tokenising the corpus, and building input-target sequences."""
import re
from collections import Counter

import torch

from . import config

UNK = "<unk>"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|--|[.,!?;:()\"]")


def load_text(path=config.DATA_PATH):
    """Return the body of the book without the Project Gutenberg header, contents and licence."""
    raw = path.read_text(encoding="utf-8-sig")
    start = re.search(r"^CHAPTER I\.$", raw, flags=re.MULTILINE).start()
    end = raw.index("THE END", start)
    return raw[start:end]


def tokenize(text):
    """Lower-case word-level tokens; punctuation marks are kept as separate tokens."""
    text = text.lower()
    for src, dst in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("—", " -- "), ("_", "")):
        text = text.replace(src, dst)
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, tokens, min_freq=config.MIN_FREQ):
        counts = Counter(tokens)
        kept = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        self.itos = [UNK] + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk_idx = 0

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def make_sequences(ids, seq_len=config.SEQ_LEN):
    """Slide a window over the token stream: X = seq_len words, y = the word that follows."""
    data = torch.tensor(ids, dtype=torch.long)
    windows = data.unfold(0, seq_len + 1, 1)
    return windows[:, :-1].contiguous(), windows[:, -1].contiguous()


def detokenize(tokens):
    """Join tokens back into readable text."""
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:)])", r"\1", text)
    return text.replace("( ", "(")


def prepare():
    """Return everything the training and generation scripts need.

    The token stream is split contiguously (train / validation / test) before windows are
    built, so no validation or test window overlaps a training window. The vocabulary is
    built from the training split only.
    """
    tokens = tokenize(load_text())
    n_train = int(len(tokens) * config.TRAIN_FRAC)
    n_val = int(len(tokens) * config.VAL_FRAC)
    splits = {
        "train": tokens[:n_train],
        "val": tokens[n_train:n_train + n_val],
        "test": tokens[n_train + n_val:],
    }
    vocab = Vocab(splits["train"])
    ids = {name: vocab.encode(toks) for name, toks in splits.items()}
    sequences = {name: make_sequences(split_ids) for name, split_ids in ids.items()}
    return {"tokens": tokens, "splits": splits, "vocab": vocab, "ids": ids, "sequences": sequences}


def describe(data):
    """Dataset statistics reported in the README."""
    tokens, vocab = data["tokens"], data["vocab"]
    counts = Counter(tokens)
    words = [t for t in tokens if t[0].isalpha()]
    sentences = sum(counts[p] for p in ".!?")
    stats = {
        "characters": len(load_text()),
        "tokens_total": len(tokens),
        "word_tokens": len(words),
        "punctuation_tokens": len(tokens) - len(words),
        "unique_tokens": len(counts),
        "vocab_size_min_freq": len(vocab),
        "tokens_seen_once": sum(1 for c in counts.values() if c == 1),
        "approx_sentences": sentences,
        "avg_sentence_len": round(len(words) / sentences, 2),
        "most_common_words": [w for w, _ in Counter(words).most_common(15)],
    }
    for name in ("train", "val", "test"):
        split_ids = data["ids"][name]
        stats[f"{name}_tokens"] = len(split_ids)
        stats[f"{name}_sequences"] = len(data["sequences"][name][1])
        stats[f"{name}_unk_rate"] = round(split_ids.count(vocab.unk_idx) / len(split_ids), 4)
    return stats
