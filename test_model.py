import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader, Subset

from model import CNN
from params import Params

data = MNIST(root="Data/", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST(root="Data/", train=False, download=True, transform=transforms.ToTensor())

# CNN model
p = Params()
model = CNN(p)
model = model.to(p.device)

# Train data
train_data, val_data = random_split(data, [0.9,0.1])
train_loader = DataLoader(train_data, batch_size=p.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=p.batch_size, shuffle=False)

# Optimizer and loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
loss_fn = nn.CrossEntropyLoss()

# Train model 
model.train()
for epoch_idx in range(p.n_epochs):

    # Loop through training data
    epoch_loss = 0.0
    for img, labels in train_loader:
        
        # Load in batch and cast image to float32
        img = img.to(p.device) # (N,1,H,W)
        labels = labels.to(p.device)

        optimizer.zero_grad()

        output = model(img) # (N,10)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {1+epoch_idx} loss = {epoch_loss:.4f}")


print("Training finished, predicting on test examples")
n_predictions = 10
im_idx = np.random.choice(len(test_data), n_predictions, replace=False)
test_subset = Subset(test_data, indices=im_idx)
test_loader = DataLoader(test_subset, batch_size=1)

model = model.to("cpu")
model.eval()
for im, label in test_loader:

    probability =  model(im)
    class_prediction = torch.argmax(probability, dim=-1)

    plt.imshow(im.numpy().squeeze(), cmap="gray")
    plt.title(f"Prediction = {class_prediction.item()}, Target = {label.item()}")
    plt.show()

