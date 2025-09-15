from __future__ import annotations
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.probs: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.probs = probs
        self.y = y
        N = y.shape[0]
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        assert self.probs is not None and self.y is not None
        N = self.y.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1.0
        grad /= N
        return grad
