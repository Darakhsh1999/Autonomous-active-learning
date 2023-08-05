import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from model import CNN
from params import Params
from training import train, test
from data import ActiveLearningDataset


def active_learn(p: Params, dataset: ActiveLearningDataset, test_loader: DataLoader):
    
    # CNN model
    model = CNN(p)
    model = model.to(p.device)
    
    # Train data
    train_data, val_data = random_split(dataset, [0.9,0.1])
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False)
    print(f"Number of training images: {len(train_data)}")

    # Optimizer and loss function 
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Train model on currently labeled data
    train(model, p, optimizer, loss_fn, train_loader, val_loader)

    # Infere on rest data and pick N most confident predictions
    unlabeled_indices = data.get_unlabeled_indices()

    # Predict on unlabeled data
    probability_scores = torch.zeros(len(unlabeled_indices), dtype=torch.float32)
    classes = torch.zeros(len(unlabeled_indices), dtype=torch.uint8)
    model.eval()
    with torch.no_grad():
        for i, data_idx in enumerate(tqdm(unlabeled_indices)): 

            im = dataset.data[data_idx]
            im = im[None,None].to(p.device, dtype=torch.float32) / 255
            probabilities = model(im).squeeze() # (10,)
            predicted_class = torch.argmax(probabilities) # class 0-9

            probability_scores[i] = probabilities[predicted_class]
            classes[i] = predicted_class

    # Extract top k most confident predictions
    if p.confidence_based:
        top_predictions = torch.argwhere(probability_scores > p.confidence_threshold).flatten()
    else:
        top_predictions = torch.topk(probability_scores, k=p.n_new_labels)
    new_indices = list(unlabeled_indices[top_predictions.indices.numpy()])
    new_labels = classes[top_predictions.indices]

    # Update the labels 
    dataset.update_labels(new_indices, new_labels)

    # Test on static ground truth test
    test_metrics = test(model, p, test_loader, is_transformed=True)

    # Calculate misclassification rate 
    test_metrics["error_rate"], test_metrics["n_errors"] = dataset.missclassification_rate()

    return test_metrics



if __name__ == "__main__":

    experiment_name = "long_run"

    # Simulation parameters
    p = Params()

    # Load in data
    data = ActiveLearningDataset(p) # continually growing data set
    test_data = MNIST(root="Data/", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False)

    history = defaultdict(list)

    for i in range(p.n_iterations):

        # Train the model on a restricted initial data set
        metrics = active_learn(p, data, test_loader)

        print(f"Iteration {1+i}")
        pprint(metrics)

        # Update history values
        for k, v in metrics.items(): history[k].append(v)
        history["data_size"].append(len(data.indices))

        # Check if we can't do one more iteration
        if len(data.get_unlabeled_indices()) < p.n_new_labels:
            break
    
    # Store the history object
    with open(f"variables/{experiment_name}_history.dat", "wb") as f:
        pickle.dump(history, f)

    