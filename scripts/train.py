from __future__ import annotations
import argparse
from np_cnn.trainer import Config, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--subset", type=str, default="10000", help="int or 'none' for full train set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--padding", type=int, default=1)
    args = p.parse_args()
    subset = None if str(args.subset).lower() == "none" else int(args.subset)
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        subset=subset,
        seed=args.seed,
        padding=args.padding,
    )
    train(cfg)


if __name__ == "__main__":
    main()
