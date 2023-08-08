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

import matplotlib.pyplot as plt

def active_learn(p: Params, dataset: ActiveLearningDataset, test_loader: DataLoader):
    
    # Fresh CNN model
    model = CNN(p)
    model = model.to(p.device)
    
    # Train data
    train_data, val_data = random_split(dataset, [0.9,0.1])
    train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False)
    print(f"Number of labeled training images: {len(dataset)}/{p.data_size}")

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
        for i, data_idx in enumerate(tqdm(unlabeled_indices, desc="Unlabeled prediction")): 

            im = dataset.data[data_idx]
            im = im[None,None].to(p.device, dtype=torch.float32) / 255
            probabilities = model(im).squeeze() # (10,)
            predicted_class = torch.argmax(probabilities) # class 0-9

            probability_scores[i] = probabilities[predicted_class]
            classes[i] = predicted_class

    if p.use_model_prediction: # Extract top k most confident predictions
        if p.confidence_based:
            top_predictions_idx = torch.argwhere(probability_scores > p.confidence_threshold).flatten()
        else:
            top_predictions_idx = torch.topk(probability_scores, k=p.n_new_labels).indices
    else: # Extract k least confident predictions
        if p.confidence_based:
            bot_predictions_idx = torch.argwhere(probability_scores < p.confidence_threshold).flatten()
        else:
            bot_predictions_idx = torch.topk(probability_scores, k=p.n_new_labels, largest=False).indices

    new_label_indices = top_predictions_idx if p.use_model_prediction else bot_predictions_idx # 0,...,n_unlabeled
    new_indices = list(unlabeled_indices[new_label_indices.numpy()]) # {0,...,data_size}
    new_labels = classes[new_label_indices] if p.use_model_prediction else dataset.ground_truth[new_indices]

    # Update the labels 
    dataset.update_labels(new_indices, new_labels)

    # Test on static ground truth test
    test_metrics = test(model, p, test_loader, is_transformed=True)

    # Calculate misclassification rate 
    test_metrics["error_rate"], test_metrics["n_errors"] = dataset.missclassification_rate()

    return model, test_metrics



if __name__ == "__main__":

    experiment_name = "long_run3_confidence"

    # Simulation parameters
    p = Params()

    # Load in data
    data = ActiveLearningDataset(p) # continually growing data set
    test_data = MNIST(root="Data/", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=p.batch_size, shuffle=False)

    history = defaultdict(list)

    for i in range(p.n_iterations):

        # Train the model on a restricted initial data set
        model, metrics = active_learn(p, data, test_loader)

        print(f"Iteration {1+i}")
        pprint(metrics)

        # Update history values
        for k, v in metrics.items(): history[k].append(v)
        history["data_size"].append(len(data.indices))

        # Check if we ran out of unlabeled data
        if len(data.get_unlabeled_indices()) == 0:
            break
        else:
            if (not p.confidence_based) and (len(data.get_unlabeled_indices()) < p.n_new_labels):
                p.n_new_labels = len(data.get_unlabeled_indices())
    
    # Detect missclassified labels
    if p.use_model_prediction:

        # Get index of missclassified data
        gt = data.ground_truth[data.indices]
        my_labels = data.targets[data.indices]
        missclassified_idx = torch.argwhere(gt != my_labels).flatten()

        # Predict on missclassified data
        new_predictions = torch.zeros(len(missclassified_idx), dtype=torch.uint8)
        with torch.no_grad():
            for i, miss_idx in enumerate(missclassified_idx):

                im = data.data[miss_idx]
                im = im[None,None].to(p.device, dtype=torch.float32) / 255
                probabilities = model(im).squeeze() # (10,)
                new_predictions[i] = torch.argmax(probabilities) # class 0-9

        # check if it the same as ground truth
        detection_rate = (new_predictions == gt[missclassified_idx]).sum() / len(missclassified_idx)
        print(f"From {len(missclassified_idx)} misslabeled data points, detected {detection_rate*100:.2f} %")
    
    # Store the history object
    save_name = f"{experiment_name}{'_modelpred' if p.use_model_prediction else ''}{'_confidence' if p.confidence_based else ''}"
    with open(f"variables/{save_name}_history.dat", "wb") as f:
        pickle.dump(history, f)

    