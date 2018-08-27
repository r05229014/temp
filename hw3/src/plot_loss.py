import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os 

dir = glob('./results_*')

for d in dir:
    iter = np.arange(0,52300)
    dd = d+'/_loss_D.npy'
    gg = d+'/_loss_G.npy'
    f = np.load(dd)
    f2 = np.load(gg)
    plt.figure()
    plt.xlim(-3,20000)
    plt.ylim(-1,28)
    plt.plot(iter, f2, label='Generator_loss')
    plt.plot(iter, f, label='Discriminator_loss')
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.legend()
    plt.savefig(d+'/loss_fig.png')
    plt.show()
