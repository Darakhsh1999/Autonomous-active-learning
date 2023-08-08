import random
import torch
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor

from params import Params

class ActiveLearningDataset(Dataset):

    def __init__(self, p: Params):
        mnist_data = MNIST(root="Data/", train=True, download=True)
        self.data = mnist_data.data[:p.data_size] # torch-uint8 (D,28,28)
        self.ground_truth = mnist_data.targets[:p.data_size].type(torch.uint8) # torch-uint8 (D,)
        initial_indices: list = random.sample(range(len(self.data)), p.initial_size)
        self.indices = initial_indices
        self.targets = torch.zeros(len(self.data), dtype=torch.uint8) # torch-uint8 (D,)
        self.targets[initial_indices] = mnist_data.targets[initial_indices].type(torch.uint8) # Write initial ground truth
    
    def update_labels(self, new_indices: np.ndarray, new_labels: torch.Tensor):
        """ Updates the dataset with the new labels """
        print(f"Added {len(new_indices)} new labels to the data set")
        self.indices += new_indices
        self.targets[new_indices] = new_labels
    
    def get_unlabeled_indices(self):
        all_indices = np.arange(len(self.data))
        return np.setdiff1d(all_indices, self.indices)
    
    def missclassification_rate(self):
        """ Calculates the current missclassification rate """

        gt = self.ground_truth[self.indices]
        my_labels = self.targets[self.indices]

        n_error = (gt != my_labels).sum()
        error_rate = n_error/len(my_labels)
        return error_rate.item(), n_error.item()
    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.data[idx], self.targets[idx] # (28,28), (,)

