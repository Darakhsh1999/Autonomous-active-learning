import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import CNN
from params import Params

def train(model: CNN, p: Params, optimizer, loss_fn, train_loader, val_loader):

    model.train()
    for _ in range(p.n_epochs):

        # Loop through training data
        epoch_loss = 0.0
        for img, labels in train_loader:

            # Load in batch and cast image to float32
            img = img[:,None].to(p.device, dtype=torch.float32) / 255 # (N,1,H,W)
            labels = labels.to(p.device)

            optimizer.zero_grad()

            output = model(img) # (N,10)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)

        # Validation score
        validation_metrics = test(model, p, val_loader)
        # Early stoppage
        p.stopping_criteria(model, validation_metrics[p.optim_metric])

    # Load in the best model w.r.t validation metric
    print(f"Training finished, best validation {p.optim_metric} = {p.stopping_criteria.best_score:.4f}")
    p.stopping_criteria.load_best_model(model)
    p.stopping_criteria.reset()


def test(model, p, test_loader: DataLoader, is_transformed=False):

    model.eval()
    class_predictions = np.zeros(len(test_loader.dataset), dtype=np.uint8)
    targets = np.zeros(len(test_loader.dataset), dtype=np.uint8)
    ptr = 0
    with torch.no_grad():
        for (img, labels) in test_loader:

            if is_transformed:
                img = img.to(p.device)
            else:
                img = img[:,None].to(p.device, dtype=torch.float32) / 255
            output_probability = model(img) # (N,10)

            predicted_batch_class = torch.argmax(output_probability, dim=-1) # (N,) class 0-9
            predicted_batch_class = predicted_batch_class.cpu().numpy().astype(np.uint8)
            for i, class_i in enumerate(predicted_batch_class):
                class_predictions[ptr] = class_i
                targets[ptr] = labels[i]
                ptr += 1

    # Calculate evaluation metrics
    accuracy = accuracy_score(targets, class_predictions)
    precision = precision_score(targets, class_predictions, average="macro", zero_division=0.0)
    recall = recall_score(targets, class_predictions, average="macro", zero_division=0.0)
    f1 = f1_score(targets, class_predictions, average="macro", zero_division=0.0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1
    }

    return metrics



