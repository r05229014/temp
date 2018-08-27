from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os 
import pandas as pd 
from skimage import io, transform
import matplotlib.pyplot as plt
# Setting hyperparameters

BATCH_SIZE = 64 
IMAGE_SIZE = 64
context_size = 2
embedding_dim = 30
raw_text = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
'green hair', 'red hair', 'purple hair', 'pink hair',
'blue hair', 'black hair', 'brown hair', 'blonde hair',
'gray eyes', 'black eyes', 'orange eyes',
'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

vocab = set()
for i in raw_text:
    vocab.update(i.split())

save_dir = "./results_3-2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Loading the dataset 
tag = pd.read_csv('../extra_data/tags.csv', header=None)

class anime_tag_Dataset(Dataset):
    """Anime tag dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with tags.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tag_csv = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.tag_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                str(self.tag_csv.iloc[idx, 0])+'.jpg')
        image = io.imread(img_name)
        tags = self.tag_csv.iloc[idx, 1]
        sample = {'image': image, 'tags': tags}

        if self.transform:
            sample = self.transform(sample)

        return sample

    

# Creating the transformations
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, tags = sample['image'], sample['tags']
        tags = [word_to_idx[w] for w in tags.split()]
        tags = np.array(tags)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'tags': torch.from_numpy(tags)}



transformed_dataset = anime_tag_Dataset(csv_file='../extra_data/tags.csv',
                                     root_dir='../extra_data/images/',
                                    transform=transforms.Compose([
                                        ToTensor()
                                    ]))
dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                       shuffle=True, num_workers=8)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class conditional_Generate(nn.Module):

    def __init__(self):
        super(conditional_GAN, self).__init__()
        self.x = nn.Sequential(
                    
                )
        self.c = nn.Sequential(
                
                )



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

netD = D()
netD.cuda()
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))


for epoch in range(300):
    
    for i, data in enumerate(dataloader, 0):

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

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 300, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 ==0:
            vutils.save_image(real, '%s/real_samples.png' % save_dir, normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (save_dir, epoch), normalize = True)

