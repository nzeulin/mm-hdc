import torch
from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.name = "mnist_example"

    config.dataset = ConfigDict()
    config.dataset.name = "mnist"
    config.dataset.num_classes = 10
    config.dataset.model_dim = 5_000

    config.training = ConfigDict()
    config.training.shuffle = True
    config.training.batch_size = 1_000
    config.training.num_epochs = 50
    config.training.eval_every = 1
    config.training.num_experiments = 2

    config.model = ConfigDict()
    config.model.learning_rate = 1e-5
    config.model.C = 500.0
    config.model.normalize = True
    config.model.transform_batch_size = None
    config.model.transform_dtype = None
    config.model.backend = "cpp"

    config.paths = ConfigDict()
    config.paths.results = "results/examples"

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.dtype = torch.float32

    return config
