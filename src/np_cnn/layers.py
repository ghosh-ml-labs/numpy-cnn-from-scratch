from __future__ import annotations
import math
from typing import Iterable, Tuple
import numpy as np


class Module:
    def params_and_grads(self):  # type: ignore[override]
        return []

    def forward(self, x: np.ndarray) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:  # type: ignore[override]
        raise NotImplementedError


class ReLU(Module):
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.mask is not None
        return grad * self.mask


class Flatten(Module):
    def __init__(self):
        self.shape: Tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.shape is not None
        return grad.reshape(self.shape)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        limit = math.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.x is not None
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params_and_grads(self):
        yield (self.W, self.dW)
        yield (self.b, self.db)


class Conv2D(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        k = kernel_size
        limit = math.sqrt(6 / (in_channels * k * k + out_channels * k * k))
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, k, k)).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self.stride, self.padding = stride, padding
        self.x: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        p = self.padding
        if p == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        N, C, H, W = x.shape
        K, _, kH, kW = self.W.shape
        H_out = (H + 2 * self.padding - kH) // self.stride + 1
        W_out = (W + 2 * self.padding - kW) // self.stride + 1
        x_p = self._pad(x)
        out = np.zeros((N, K, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for k in range(K):
                for i in range(H_out):
                    hs = i * self.stride
                    for j in range(W_out):
                        ws = j * self.stride
                        region = x_p[n, :, hs : hs + kH, ws : ws + kW]
                        out[n, k, i, j] = np.sum(region * self.W[k]) + self.b[k]
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.x is not None
        x, W = self.x, self.W
        N, C, H, W_in = x.shape
        K, _, kH, kW = W.shape
        x_p = self._pad(x)
        dx_p = np.zeros_like(x_p)
        self.dW.fill(0.0)
        self.db = grad.sum(axis=(0, 2, 3))
        H_out, W_out = grad.shape[2], grad.shape[3]
        for n in range(N):
            for k in range(K):
                for i in range(H_out):
                    hs = i * self.stride
                    for j in range(W_out):
                        ws = j * self.stride
                        region = x_p[n, :, hs : hs + kH, ws : ws + kW]
                        g = grad[n, k, i, j]
                        self.dW[k] += g * region
                        dx_p[n, :, hs : hs + kH, ws : ws + kW] += g * W[k]
        p = self.padding
        return dx_p[:, :, p : H + p, p : W_in + p] if p > 0 else dx_p

    def params_and_grads(self):
        yield (self.W, self.dW)
        yield (self.b, self.db)


class MaxPool2D(Module):
    def __init__(self, kernel_size: int, stride: int):
        self.k = kernel_size
        self.s = stride
        self.argmax: dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
        self.x_shape: Tuple[int, int, int, int] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        H_out = (H - self.k) // self.s + 1
        W_out = (W - self.k) // self.s + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        self.argmax.clear()
        self.x_shape = x.shape
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    hs = i * self.s
                    for j in range(W_out):
                        ws = j * self.s
                        window = x[n, c, hs : hs + self.k, ws : ws + self.k]
                        idx = np.argmax(window)
                        out[n, c, i, j] = window.flat[idx]
                        self.argmax[(n, c, i, j)] = (hs + idx // self.k, ws + idx % self.k)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self.x_shape is not None
        N, C, H, W = self.x_shape
        dx = np.zeros((N, C, H, W), dtype=np.float32)
        H_out, W_out = grad.shape[2], grad.shape[3]
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        r, s = self.argmax[(n, c, i, j)]
                        dx[n, c, r, s] += grad[n, c, i, j]
        return dx
