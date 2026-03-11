from absl import app, flags
from ml_collections.config_flags import config_flags
import numpy as np
import torch
import os
from typing import Any, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from importlib.util import spec_from_file_location, module_from_spec
from data import load_mnist
from ml_collections import ConfigDict
from tqdm import tqdm

import hdc
from hdc.mmhdc import MultiMMHDC
from hdc import HDTransform

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Path to configuration.")

def save_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    experiment: int,
                    epoch: int,
                    config: ConfigDict):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(
        checkpoint_path, f"{config.name}_exp{experiment}_epoch{epoch}.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)

def evaluate_model(model, X_test: torch.Tensor, y_test: torch.Tensor):
    """Evaluate model and return metrics."""
    with torch.no_grad():
        y_pred = model(X_test)
        accuracy = (y_pred == y_test).float().mean().item()

        return {
            'accuracy': accuracy,
            # 'loss': loss,
        }

def load_model_config(config_path: str):
    """Load model specific config."""
    spec = spec_from_file_location("model_config", config_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


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

def main(_):
    torch.random.manual_seed(0)
    np.random.seed(0)

    config = FLAGS.config
    if config is None:
        raise ValueError("Configuration file must be provided.")

    checkpoints_dir = config.paths.checkpoints
    checkpoints_meta_dir = config.paths.checkpoints_meta
    results_dir = config.paths.results
    dtype = config.dtype

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(checkpoints_meta_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_mnist(config.dataset.name)
    X_train = torch.tensor(X_train, dtype=dtype)
    X_test = torch.tensor(X_test, dtype=dtype)

    lbl_enc = LabelEncoder().fit(y_train)
    y_train = torch.tensor(lbl_enc.transform(y_train), dtype=torch.int64)
    y_test = torch.tensor(lbl_enc.transform(y_test), dtype=torch.int64)

    # Load model config (single model only)
    assert len(config.model_config_paths) == 1, "Only one model config is supported."
    model_config = load_model_config(config.model_config_paths[0])

    tracking = {}
    for exp_i in range(config.training.num_experiments):
        experiment_results = []

        # HD transform
        X_train_flat = X_train.view(X_train.shape[0], -1)
        X_test_flat = X_test.view(X_test.shape[0], -1)
        hd_transform = HDTransform(
            X_train_flat.shape[1],
            out_channels=config.dataset.model_dim,
            batch_size=1024,
            seed=exp_i,
            batch_norm=False,
            device=config.device,
            transform_type=config.dataset.mapping,
            dtype=dtype,
        )
        X_train_hd = hd_transform(X_train_flat)
        X_test_hd = hd_transform(X_test_flat).to(config.device)
        y_train_exp = y_train.clone()
        y_test_exp = y_test.clone().to(config.device)

        # Initialize the model
        model = MultiMMHDC(
            out_channels=config.dataset.model_dim,
            num_classes=config.dataset.num_classes,
            lr=model_config.learning_rate,
            C=model_config.C,
            device=config.device,
            backend=model_config.backend,
            dtype=dtype,)
        model.initialize(X_train_hd, y_train_exp)

        # Training loop
        rng = torch.manual_seed(exp_i)
        X_train_hd_shuf = X_train_hd.clone()
        y_train_shuf = y_train_exp.clone()
        for epoch in tqdm(range(config.training.num_epochs)):
            epoch_metrics = {'epoch': epoch}

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

            if epoch % config.training.checkpoint_every == 0:
                save_checkpoint(checkpoints_meta_dir, model, exp_i, epoch, config)

        tracking[f'experiment_{exp_i}'] = experiment_results
        save_checkpoint(checkpoints_dir, model, exp_i, epoch, config)

    # Save final results
    torch.save(tracking, os.path.join(results_dir, '{}_experiment_results.pt'.format(config.name)))

if __name__ == "__main__":
    app.run(main)
