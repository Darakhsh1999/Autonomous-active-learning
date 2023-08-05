import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, p):
        super().__init__()

        var = p.var
        kernel_size = p.kernel_size

        self.conv1 = nn.Conv2d(1, 4*var, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(4*var, 8*var, kernel_size=kernel_size, padding="same")
        self.conv3 = nn.Conv2d(8*var, 16*var, kernel_size=kernel_size, padding="same")
        self.conv4 = nn.Conv2d(16*var, 32*var, kernel_size=kernel_size, padding="same")
        self.linear1 = nn.Linear(32*var*7*7, 100)
        self.linear2 = nn.Linear(100, p.n_classes)
        self.soft_max = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(kernel_size=p.maxpool_kernel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = torch.flatten(x, start_dim= 1, end_dim= -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.soft_max(x)
        return x