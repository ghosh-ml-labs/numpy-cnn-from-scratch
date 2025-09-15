from __future__ import annotations
import numpy as np


def seed_everything(seed: int = 42):
    np.random.seed(seed)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())
