import torch
from ml_collections import ConfigDict
from torch.cuda import is_available as cuda_available
import os

def get_config():
    """Returns the default configuration for the MNIST dataset."""
    config = ConfigDict()
    config.name = 'default_mnist_config'

    # Dataset configuration
    config.dataset = ConfigDict()
    dataset = config.dataset
    dataset.name = 'mnist'
    dataset.mapping = 'onlinehd'  # HDC mapping
    dataset.num_classes = 10
    dataset.model_dim = 5_000

    # Training configuration
    config.training = ConfigDict()
    training = config.training
    training.shuffle = True
    training.batch_size = 1000
    training.num_epochs = 50
    training.eval_every = 1  # Evaluate every N epochs
    training.checkpoint_every = 2  # Save checkpoint every N epochs
    training.num_experiments = 2

    # Paths configuration
    config.paths = ConfigDict()
    paths = config.paths
    paths.checkpoints = 'checkpoints'
    paths.checkpoints_meta = 'checkpoints_meta'
    paths.results = 'results'
    paths.configs = 'configs'

    # General configuration
    config.model_config_paths = [
        os.path.join(paths.configs, dataset.name, "hdc/mmhdc_multi_config.py"),
    ]

    # Device
    config.device = 'cuda'
    config.dtype = torch.float32

    return config
