import torch
from early_stoppage import EarlyStopping

class Params():

    # Data set parameters
    data_size = 10000
    initial_size = 1000

    # Training parameters
    n_epochs = 40
    lr = 0.001
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    optim_metric = "accuracy"
    optim_mode = "max"
    stopping_criteria = EarlyStopping(mode=optim_mode)

    # Model architecture
    var = 1
    kernel_size = (3,3)
    maxpool_kernel = (2,2)

    # Active learning parameters
    n_iterations = 20
    n_new_labels = 100
    confidence_based = True # If True, we add prediction's label if probability score is above confidence_threshold
    confidence_threshold = 0.99999