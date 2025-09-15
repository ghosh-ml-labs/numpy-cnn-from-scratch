from __future__ import annotations
from typing import Tuple, Iterator
import numpy as np


def load_mnist_numpy(limit: int | None = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_train, y_train, x_test, y_test) with shapes (N,1,28,28)."""
    try:
        import torchvision  # type: ignore
        from torchvision import transforms  # type: ignore

        tfm = transforms.Compose([transforms.ToTensor()])
        train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
        x_train = train.data.numpy().astype(np.float32) / 255.0
        y_train = train.targets.numpy().astype(np.int64)
        x_test = test.data.numpy().astype(np.float32) / 255.0
        y_test = test.targets.numpy().astype(np.int64)
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]
    except Exception:
        from sklearn.datasets import fetch_openml  # type: ignore

        mnist = fetch_openml(name="mnist_784", version=1, as_frame=False)
        X = (mnist.data.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
        y = mnist.target.astype(np.int64)
        x_train, y_train = X[:60000], y[:60000]
        x_test, y_test = X[60000:], y[60000:]

    if limit is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(x_train), size=limit, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
    return x_train, y_train, x_test, y_test


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
