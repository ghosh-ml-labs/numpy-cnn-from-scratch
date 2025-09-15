from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .utils import seed_everything, accuracy
from .layers import Conv2D, MaxPool2D, ReLU, Flatten, Linear
from .losses import CrossEntropyLoss
from .optim import SGD
from .model import Sequential
from .data import load_mnist_numpy, iterate_minibatches


@dataclass
class Config:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.05
    momentum: float = 0.9
    subset: int | None = 10000
    seed: int = 42
    padding: int = 1


def build_model(cfg: Config) -> tuple[Sequential, CrossEntropyLoss, SGD]:
    model = Sequential(
        Conv2D(1, 8, kernel_size=3, stride=1, padding=cfg.padding),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Conv2D(8, 16, kernel_size=3, stride=1, padding=cfg.padding),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Flatten(),
        Linear(16 * 7 * 7, 10),
    )
    loss = CrossEntropyLoss()
    opt = SGD([model], lr=cfg.lr, momentum=cfg.momentum)
    return model, loss, opt


def train(cfg: Config):
    seed_everything(cfg.seed)
    Xtr, Ytr, Xte, Yte = load_mnist_numpy(limit=cfg.subset, seed=cfg.seed)
    model, loss_fn, opt = build_model(cfg)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for xb, yb in iterate_minibatches(Xtr, Ytr, cfg.batch_size, shuffle=True):
            out = model.forward(xb)
            l = loss_fn.forward(out, yb)
            grad_logits = loss_fn.backward()
            model.backward(grad_logits)
            opt.step()
            train_loss += l * xb.shape[0]
            train_correct += (out.argmax(axis=1) == yb).sum()
            train_count += xb.shape[0]
        train_acc = float(train_correct) / float(train_count)
        train_loss /= float(train_count)

        # quick eval on a fixed-size subset for speed
        eval_idx = np.random.choice(len(Xte), size=min(5000, len(Xte)), replace=False)
        Xe, Ye = Xte[eval_idx], Yte[eval_idx]
        logits = model.forward(Xe)
        eval_loss = loss_fn.forward(logits, Ye)
        eval_acc = accuracy(logits, Ye)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={eval_loss:.4f} acc={eval_acc:.4f}")
