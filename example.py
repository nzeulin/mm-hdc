from absl import app, flags
from ml_collections.config_flags import config_flags
import os
from typing import Dict
import torch
from data import load_mnist
from tqdm import tqdm

from mmhdc import MultiMMHDC
from mmhdc.utils import HDTransform

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to configuration.")

def evaluate_model(model, X_test: torch.Tensor, y_test: torch.Tensor):
    """Evaluate model and return metrics."""
    with torch.no_grad():
        y_pred = model(X_test)
        loss = model.loss(X_test, y_test).item()
        accuracy = (y_pred == y_test).float().mean().item()

        return {
            'accuracy': accuracy,
            'loss': loss,
        }

def print_metrics_summary(epoch: int, experiment: int, metrics: Dict[str, Any]):
    """Print evaluation metrics."""
    print(f"\nExp {experiment} Epoch {epoch} Evaluation Summary:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        if value is not None:
            if isinstance(value, list):
                formatted_list = [f"{x:.4f}" for x in value]
                print(f"{metric_name}={formatted_list} ", end="")
            else:
                print(f"{metric_name}={value:.4f} ", end="")
    print("\n" + "-" * 50)

def load_dataset(config):
    X_train, y_train, X_test, y_test = load_mnist(config.dataset.name)
    X_train = X_train.to(dtype=config.dtype)
    X_test = X_test.to(dtype=config.dtype)
    return X_train, y_train, X_test, y_test


def run_experiment(config, exp_i: int, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_test: torch.Tensor, y_test: torch.Tensor):
    experiment_results = []
    transform_dtype = config.model.transform_dtype or config.dtype

    X_train_flat = X_train.view(X_train.shape[0], -1)
    X_test_flat = X_test.view(X_test.shape[0], -1)
    hd_transform = HDTransform(
        X_train_flat.shape[1],
        out_channels=config.dataset.model_dim,
        batch_size=config.model.transform_batch_size,
        seed=exp_i,
        normalize=bool(config.model.normalize),
        device=config.device,
        dtype=transform_dtype,
    )
    X_train_hd = hd_transform(X_train_flat)
    X_test_hd = hd_transform(X_test_flat).to(config.device)
    y_train_exp = y_train.clone()
    y_test_exp = y_test.clone().to(config.device)

    model = MultiMMHDC(
        out_channels=config.dataset.model_dim,
        num_classes=config.dataset.num_classes,
        lr=config.model.learning_rate,
        C=config.model.C,
        device=config.device,
        backend=config.model.backend,
        dtype=config.dtype,
    )
    model.initialize(X_train_hd, y_train_exp)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(exp_i)
    X_train_hd_shuf = X_train_hd.clone()
    y_train_shuf = y_train_exp.clone()
    for epoch in tqdm(range(config.training.num_epochs)):
        epoch_metrics = {"epoch": epoch}

        if config.training.shuffle:
            idx = torch.randperm(X_train_hd_shuf.shape[0], generator=rng)
            X_train_hd_shuf = X_train_hd_shuf[idx]
            y_train_shuf = y_train_shuf[idx]

        for i in range(0, X_train_hd_shuf.shape[0], config.training.batch_size):
            batch_X = X_train_hd_shuf[i:i + config.training.batch_size].to(model.device)
            batch_y = y_train_shuf[i:i + config.training.batch_size].to(model.device)
            model.step(batch_X, batch_y)

        if epoch % config.training.eval_every == 0:
            metrics = evaluate_model(model, X_test_hd, y_test_exp)
            epoch_metrics.update(metrics)
            experiment_results.append(epoch_metrics)
            print_metrics_summary(epoch, exp_i, metrics)

    return experiment_results


def main(_):
    torch.manual_seed(0)

    config = FLAGS.config
    if config is None:
        raise ValueError("Configuration file must be provided.")

    results_dir = config.paths.results
    dtype = config.dtype
    os.makedirs(results_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset(config)
    tracking = {}
    for exp_i in range(config.training.num_experiments):
        tracking[f"experiment_{exp_i}"] = run_experiment(config, exp_i, X_train, y_train, X_test, y_test)

    torch.save(tracking, os.path.join(results_dir, f"{config.name}_experiment_results.pt"))

if __name__ == "__main__":
    app.run(main)
