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
from torchsummary import summary
from tqdm import tqdm
import os
import numpy as np
# Setting hyperparameters

BATCH_SIZE = 64 
IMAGE_SIZE = 64

# Creating the transformations

transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(), 
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    ])

# Loading the dataset 
dataset = dset.ImageFolder(root = '../AnimeDataset/', transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 8)


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G_o(nn.Module):

    def __init__(self):
        super(G_o, self).__init__()
        self.main = nn.Sequential(
                    nn.Linear(100,128),
                    nn.ReLU(True),
                    nn.Linear(128,256),
                    nn.ReLU(True),
                    nn.Linear(256,256),
                    nn.ReLU(True),
                    nn.Linear(256,256),
                    nn.ReLU(True),
                    nn.Linear(256,12288),              
                    nn.Tanh()
                ) 
    def forward(self, input):
        output = self.main(input)
        return output.view(-1,3,64,64)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
                    nn.ConvTranspose2d(100,512,4,1,0, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512,256,4,2,1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256,128,4,2,1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128,64,4,2,1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64,3,4,2,1, bias =False),
                    nn.Tanh()
                ) 
    def forward(self, input):
        output = self.main(input)
        return output



class D_o(nn.Module):

    def __init__(self):
        super(D_o,self).__init__()
        self.main = nn.Sequential(
                    nn.Linear(3*64*64,128),              
                    nn.ReLU(True),
                    nn.Linear(128,64),              
                    nn.ReLU(True),
                    nn.Linear(64,32),              
                    nn.ReLU(True),
                    nn.Linear(32,16),              
                    nn.ReLU(True),
                    nn.Linear(16,1),              
                    nn.Sigmoid()
                )

    def forward(self, input):
        output = self.main(input.view(-1, 3*64*64))
        return output.view(-1)


class D(nn.Module):

    def __init__(self):
        super(D,self).__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(3,64,4,2,1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64,128,4,2,1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128,256,4,2,1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256,512,4,2,1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(512,1,4,1,0, bias=False),
                    nn.Sigmoid()
                )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)



netG = G()
netG.cuda()
netG.apply(weights_init)
summary(netG,input_size=(100,1,1))
print(netG)

netD = D()
netD.cuda()
netD.apply(weights_init)
summary(netD,input_size=(3,64,64,))
print(netD)


criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))
#optimizerD = optim.SGD(netD.parameters(), lr = 0.01, momentum=0.9)
#optimizerG = optim.SGD(netG.parameters(), lr = 0.01, momentum=0.9) 

loss_D = []
loss_G = []
dir = "./results_dc"
for epoch in range(100):
    
    for i, data in tqdm(enumerate(dataloader, 0)):
        # Train Discriminator
        netD.zero_grad()

        real, _ = data # real data
        input = Variable(real).cuda()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(input)
        errD_real = criterion(output, target)

        noise = Variable(torch.randn(input.size()[0], 100,1,1)).cuda() # fake data
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0])).cuda()
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        loss_D.append(errD.data[0])
        loss_G.append(errG.data[0])

        #print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 300, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 ==0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 100, i, len(dataloader), errD.data[0], errG.data[0]))
            vutils.save_image(real, '%s/real_samples.png' % dir, normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (dir, epoch), normalize = True)

loss_D = np.array(loss_D)
loss_G = np.array(loss_G)
np.save(dir+'/_loss_D', loss_D)
np.save(dir+'/_loss_G', loss_G)

