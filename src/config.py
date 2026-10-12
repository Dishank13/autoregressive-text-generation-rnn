"""Shared experimental settings. Every model is trained with exactly these values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "alice_in_wonderland.txt"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

MODEL_NAMES = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]

# Data
SEQ_LEN = 20          # number of context words used to predict the next word
MIN_FREQ = 2          # words seen fewer times are mapped to <unk>
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1        # the remaining 10% is the test split

# Model
EMBED_DIM = 100
HIDDEN_DIM = 128      # per direction for the bidirectional model
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
GRAD_CLIP = 5.0
SEED = 42

# Generation
TEMPERATURES = [0.2, 0.7, 1.0, 1.5]
NUM_SEEDS = 3         # seed sequences taken from the held-out test split
GEN_LEN = 100         # words generated per sample
SAMPLES_PER_SETTING = 5
