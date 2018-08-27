from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os 
# Setting hyperparameters

BATCH_SIZE = 64 
IMAGE_SIZE = 64

save_dir = "./results_3-2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Creating the transformations

transform = transforms.Compose([
                transforms.Scale(IMAGE_SIZE),
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    ])

# Loading the dataset 
dataset = dset.ImageFolder(root = '../extra_data', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)

