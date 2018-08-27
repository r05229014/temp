import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import torch 
from PIL import Image
import cv2

data_dir = '../AnimeDataset/faces/'
save_dir = '../Data/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_list = os.listdir(data_dir)
 
#print(img_list)

transform1 = transforms.Compose([
                transforms.ToTensor(),
                ])

img1 = cv2.imread(data_dir + img_list[0])
print(img1)
print(img1.shape)
cv2.imshow('img1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = transform1(img1)
img2 = img2.numpy()*255
img2 = img2.astype('uint8')
print(img2.shape)
img2 = np.transpose(img2, (1,2,0))

print(img2.shape)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

transforms4 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomCrop((30,30)),
    ])
img3 = Image.open(data_dir + img_list[0]).convert('RGB')
img3.show()
img3 = transforms4(img3)
img3.show()
