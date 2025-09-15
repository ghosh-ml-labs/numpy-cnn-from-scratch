# numpy-cnn-from-scratch


A minimal Convolutional Neural Network implemented **from scratch in NumPy**. Includes Conv2D, MaxPool2D, ReLU, Flatten, and Linear layers with backpropagation, a stable softmax cross-entropy loss, and an SGD optimizer with momentum. Trains on a **MNIST subset** for speed and correctness-first learning.


## Features
- Manual layers with forward/backward
- Softmax Cross-Entropy (numerically stable)
- SGD + Momentum
- MNIST loader (prefers `torchvision`, falls back to `sklearn` OpenML)
- Simple training loop with minibatches and accuracy metrics
- Pre-commit hooks (Black, Ruff, isort) and GitHub Actions CI


## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --epochs 5 --subset 10000 --batch-size 128 --lr 0.05
```


## Project Structure
```
numpy-cnn-from-scratch/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/
│ └── np_cnn/
│ ├── __init__.py
│ ├── utils.py
│ ├── layers.py
│ ├── losses.py
│ ├── optim.py
│ ├── data.py
│ ├── model.py
│ └── trainer.py
└── scripts/
└── train.py
```


## Results (reference)
- MNIST (10k subset): ≥95% accuracy in a few epochs on CPU


## License
MIT


---
