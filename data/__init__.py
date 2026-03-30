from pathlib import Path
import torch
from torchvision import datasets

def load_mnist(dataset: str = "mnist"):
    dataset_map = {
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }

    dataset_key = dataset.lower()
    if dataset_key not in dataset_map:
        valid = ", ".join(sorted(dataset_map))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {valid}")

    dataset_cls = dataset_map[dataset_key]
    data_root = Path(__file__).resolve().parent

    train_set = dataset_cls(root=str(data_root), train=True, download=True)
    test_set = dataset_cls(root=str(data_root), train=False, download=True)

    X_train = train_set.data.to(dtype=torch.float32)
    y_train = train_set.targets.to(dtype=torch.int64)
    X_test = test_set.data.to(dtype=torch.float32)
    y_test = test_set.targets.to(dtype=torch.int64)

    return X_train, y_train, X_test, y_test
