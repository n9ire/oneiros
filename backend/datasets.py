"""
Dataset loading for built-in PyTorch datasets.
All data is downloaded to ./data/ on first use.
"""

from __future__ import annotations

import ssl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

# macOS Python.org installs lack system CA certificates by default, causing SSL errors
# when torchvision tries to download datasets. Permanent fix for your machine:
#   open /Applications/Python\ 3.13/Install\ Certificates.command
# This line handles it automatically for local dev:
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Dataset configs ───────────────────────────────────────────────────────────

_DATASET_REGISTRY: dict[str, dict] = {
    "mnist": {
        "cls": datasets.MNIST,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "name": "MNIST",
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
        "input_shape": (1, 28, 28),
        "num_classes": 10,
        "name": "Fashion-MNIST",
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]),
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "name": "CIFAR-10",
    },
}


def get_dataset_info(name: str) -> dict | None:
    cfg = _DATASET_REGISTRY.get(name)
    if cfg is None:
        return None
    return {
        "input_shape": cfg["input_shape"],
        "num_classes": cfg["num_classes"],
        "name": cfg["name"],
    }


def list_datasets() -> list[str]:
    return list(_DATASET_REGISTRY.keys())


def get_loaders(
    name: str,
    batch_size: int = 64,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    Downloads data on first call.
    """
    cfg = _DATASET_REGISTRY.get(name)
    if cfg is None:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    DatasetCls = cfg["cls"]
    transform = cfg["transform"]

    full_train = DatasetCls(DATA_DIR, train=True, download=True, transform=transform)
    test_set = DatasetCls(DATA_DIR, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    num_workers = 0  # 0 = main process (safe for all platforms)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Test set kept for later (checkpoints / eval endpoint)
    _ = test_set

    return train_loader, val_loader


# ── Custom tabular dataset ────────────────────────────────────────────────────

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X: list[list[float]], y: list[int]) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Reshape to (C, 1, 1) so the model's flatten layer gives (C,)
        x = self.X[idx].unsqueeze(-1).unsqueeze(-1)
        return x, self.y[idx]


def make_custom_loaders(
    payload: dict,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    train_set = TabularDataset(payload["X_train"], payload["y_train"])
    val_set = TabularDataset(payload["X_val"], payload["y_val"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader
