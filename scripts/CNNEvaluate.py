import numpy as np
import copy
import time
import sys
sys.path.append('scripts')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from torch.utils import data
import h5py
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('../CaloChallenge/code')
sys.path.append('CaloChallenge/code')
from utils import *
from models import *
from XMLHandler import *





class CNN(nn.Module):
    '''UNet to evaluate the performance of the model'''
    def __init__(
        self,
        out_dim=1,
        layer_sizes=None,
        channels=1,
        cond_dim=64,
        cylindrical=False,
        data_shape=(-1, 1, 45, 16, 9),
        NN_embed=None,
        device = 'cpu',
        RZ_shape=None,
        mid_attn = True
    ):
        super().__init__()

        self.NN_embed=NN_embed
        for param in self.NN_embed.parameters():
            param.requires_grad = False


        # determine dimensions
        self.channels=channels
        self.mid_attn=mid_attn
        self.device = device

        self.R_image, self.Z_image = create_R_Z_image(self.device, scaled=True, shape=RZ_shape)
        self.phi_image = create_phi_image(self.device, shape=RZ_shape)

        if(not cylindrical): self.init_conv=nn.Conv3d(channels, layer_sizes[0], kernel_size=3, padding=1)

        else: self.init_conv=CylindricalConv(channels, layer_sizes[0], kernel_size=3, padding=1)

        # energy embeddings
        half_cond_dim = cond_dim // 2
        cond_layers=[]

        cond_layers=[nn.Unflatten(-1, (-1, 1)), nn.Linear(1, half_cond_dim//2),nn.GELU()]
        cond_layers += [ nn.Linear(half_cond_dim//2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, cond_dim)]

        self.cond_mlp=nn.Sequential(*cond_layers)

        self.block1 = ResnetBlock(dim=layer_sizes[0], dim_out=layer_sizes[1], cond_emb_dim=cond_dim, groups=8, cylindrical=cylindrical)
        self.downsample1 = Downsample(dim = layer_sizes[1], cylindrical=cylindrical, compress_Z = True)

        self.block2 = ResnetBlock(dim=layer_sizes[1], dim_out=layer_sizes[2], cond_emb_dim=cond_dim, groups=8,
                                  cylindrical=cylindrical)
        self.downsample2 = Downsample(dim=layer_sizes[2], cylindrical=cylindrical, compress_Z=True)

        self.final_conv = nn.Sequential(ResnetBlock(dim=layer_sizes[2], dim_out=layer_sizes[3], cond_emb_dim=cond_dim, groups=8, cylindrical=cylindrical),
                                        CylindricalConv(layer_sizes[3], layer_sizes[4], 1))

        self.final_lin = nn.Sequential(nn.Linear(896, 128, device = self.device), nn.ReLU(), nn.Linear(128, 1, device = self.device))

    def forward(self, x, cond):

        x=self.NN_embed.enc(x)
        x=self.add_RZPhi(x)
        x=self.init_conv(x)


        conditions=self.cond_mlp(cond)


        x = self.block1(x,conditions)

        x = self.downsample1(x)

        x = self.block2(x, conditions)

        x = self.downsample2(x)

        x = self.final_conv(x)

        fcn = x.view(x.size(0), -1)
        return self.final_lin(fcn)

    def add_RZPhi(self, x):
        cats = [x]

        batch_R_image = self.R_image.repeat([x.shape[0], 1, 1, 1, 1]).to(device=x.device)
        batch_Z_image = self.Z_image.repeat([x.shape[0], 1, 1, 1, 1]).to(device=x.device)

        cats += [batch_R_image, batch_Z_image]

        batch_phi_image = self.phi_image.repeat([x.shape[0], 1, 1, 1, 1]).to(device=x.device)

        cats += [batch_phi_image]

        if (len(cats) > 1):
            return torch.cat(cats, axis=1)
        else:
            return x

def ttv_split(data1, data2, split=np.array([0.6, 0.2, 0.2])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    assert len(data1) == len(data2)
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    train2, test2, val2 = np.split(data2, num_events.cumsum()[:-1])
    train = np.concatenate([train1, train2], axis=0)
    test = np.concatenate([test1, test2], axis=0)
    val = np.concatenate([val1, val2], axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    return train, test, val


