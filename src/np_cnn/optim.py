from __future__ import annotations
from typing import List, Tuple, Iterable
import numpy as np

from .layers import Module


class SGD:
    def __init__(self, modules: Iterable[Module], lr: float = 1e-2, momentum: float = 0.9):
        self.lr = lr
        self.m = momentum
        self.params: List[Tuple[np.ndarray, np.ndarray]] = []
        for m in modules:
            for p, g in m.params_and_grads():
                self.params.append((p, g))
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self):
        for i, (p, g) in enumerate(self.params):
            self.v[i] = self.m * self.v[i] - self.lr * g
            p += self.v[i]
