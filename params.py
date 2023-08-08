import torch
from early_stoppage import EarlyStopping

class Params():

    verbose = 0 # 0 = no prints, 1 = iteration, 2 = metrics

    # Active learning parameters
    n_iterations = 40
    n_new_labels = 100
    use_model_prediction = False # If True, use model's prediction on unlabeled data as the new label
    confidence_based = False # If True, we add prediction's label if probability score is above confidence_threshold
    confidence_threshold = 0.99999 if use_model_prediction else 0.8

    # Data set parameters
    data_size = 10000
    initial_size = 1000

    # Training parameters
    n_epochs = 50
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
