import torch
from ml_collections import ConfigDict


def get_config():
    """Default config for the MNIST regularization-vs-dimension ablation."""
    config = ConfigDict()
    config.name = "mnist_reg_vs_dim"

    config.dataset = ConfigDict()
    config.dataset.name = "mnist"
    config.dataset.num_classes = 10

    config.training = ConfigDict()
    config.training.shuffle = True
    config.training.batch_size = 1_000
    config.training.num_epochs = 150
    config.training.num_experiments = 2

    config.model = ConfigDict()
    config.model.learning_rate = 1e-5
    config.model.normalize = True
    config.model.transform_batch_size = None
    config.model.transform_dtype = None
    config.model.backend = "cpp"

    config.sweep = ConfigDict()
    config.sweep.model_dims = [250, 1_000]
    # config.sweep.regularization_constants = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    config.sweep.regularization_constants = [1e-3, 1e0, 1e3]

    config.paths = ConfigDict()
    config.paths.results = "results/ablation"

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dtype = torch.float32

    return config
