from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = MNIST(root= "Data/", train= True, download= True, transform= transforms.ToTensor())
    return data

if __name__ == "__main__":


    train_data = load_data()
    n_images = 5

    for _ in range(n_images):

        img, digit = train_data[np.random.randint(0,len(train_data))]
        plt.imshow(img.squeeze(), cmap= "gray")
        plt.title(f"Digit {digit}")
        plt.show()