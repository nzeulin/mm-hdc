import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict()

    # Training configuration
    config.name = ''
    config.learning_rate = 1e-5
    config.normalize = True
    config.transform_batch_size = None
    config.zero_bias = True
    config.backend = 'cpp'

    return config
