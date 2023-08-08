import random
import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from model import CNN
from params import Params
from training import train, test
from data import ActiveLearningDataset
