import torch
import torch.nn as nn
from data import load_data
from torch.utils.data import random_split, DataLoader


class CNN(nn.Module):

    def __init__(self, n_classes, kernal_size= (3,3)):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 4, kernal_size)
        self.conv2 = nn.Conv2d(4, 8, kernal_size)
        self.conv3 = nn.Conv2d(8, 16, kernal_size)
        self.conv4 = nn.Conv2d(16, 32, kernal_size)
        self.linear1 = nn.Linear(32*20*20, 100)
        self.linear2 = nn.Linear(100, n_classes)
        self.soft_max = nn.Softmax(dim= -1)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim= 1, end_dim= -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.soft_max(x)
        return x

def  train_model(n_epochs, model, train_loader, loss_fn,  optimizer, device):

    for epoch_idx in range(n_epochs):
        epoch_loss = 0.0
        for batch in train_loader:

            img, labels = batch
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(img)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {1+epoch_idx}/{n_epochs}: Loss = {epoch_loss:.5f}")

def active_learn():
    
    # Train model on currently labeled data


    # Infer on rest data and pick N most confident predictions

    # add the most confident labels to our labeled data

    pass


if __name__ == "__main__":

    # CNN model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"Using device = {device}")
    model = CNN(n_classes= 10)
    model = model.to(device)

    # Optimizer and loss function 
    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Load in data
    data = load_data()
    
    # Train the model on a restricted initial data set
    s_initial = 3000
    train_epochs = 10
    initial_data, _ = random_split(data, [s_initial, len(data)-s_initial])
    train_loader = DataLoader(initial_data, batch_size= 16)
    train_model(train_epochs, model, train_loader, loss_fn, optimizer, device)
    