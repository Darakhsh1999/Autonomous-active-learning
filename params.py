import torch
from early_stoppage import EarlyStopping

class Params():

    # Training parameters
    data_size = 10000
    initial_size = 500
    n_epochs = 40
    batch_size = 64
    lr = 0.001
    var = 1
    kernel_size = (3,3)
    maxpool_kernel = (2,2)
    n_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    optim_metric = "accuracy"
    optim_mode = "max"
    stopping_criteria = EarlyStopping(mode=optim_mode)

    # Active learning parameters
    n_iterations = 20
    n_new_labels = 100
    confidence_threshold = 0.9