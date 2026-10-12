"""The four recurrent next-word models. They differ only in the recurrent layer."""
import torch
from torch import nn

from . import config

RECURRENT_LAYERS = {
    "SimpleRNN": (nn.RNN, False),
    "BiRNN": (nn.RNN, True),
    "LSTM": (nn.LSTM, False),
    "GRU": (nn.GRU, False),
}


class NextWordModel(nn.Module):
    """Embedding -> recurrent layer -> dropout -> linear layer over the vocabulary.

    The model reads a window of previous words and predicts the single word that follows.
    Because the target lies outside the window, the bidirectional variant can read the
    context in both directions without ever seeing the word it has to predict.
    """

    def __init__(self, name, vocab_size, embed_dim=config.EMBED_DIM,
                 hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT):
        super().__init__()
        rnn_cls, bidirectional = RECURRENT_LAYERS[name]
        self.name = name
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, hidden = self.rnn(emb)
        if isinstance(hidden, tuple):  # LSTM returns (hidden state, cell state)
            hidden = hidden[0]
        # (directions, batch, hidden) -> (batch, directions * hidden)
        hidden = hidden.transpose(0, 1).reshape(x.size(0), -1)
        return self.fc(self.dropout(hidden))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(name, vocab_size):
    model = NextWordModel(name, vocab_size)
    state = torch.load(config.CHECKPOINT_DIR / f"{name}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
