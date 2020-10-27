import torch
import torch.nn as nn

import json
from easydict import EasyDict as edict
from graphs.weights_initializer import weights_init
from graphs.models.custom_functions.weight_norm import WN_Conv3d, WN_ConvTranspose3d, WN_Linear

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)
        kernel_size = [5,5,5]
        stride = [2,2,2]

        self.conv1 = nn.Conv3d(in_channels=2, out_channels=128, kernel_size=kernel_size, stride=stride)
        self.batch_norm1 = nn.BatchNorm3d(128)

        self.conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride)
        self.batch_norm2 = nn.BatchNorm3d(256)

        self.conv3 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride)
        self.batch_norm3 = nn.BatchNorm3d(512)

        self.linear = WN_Linear(512, self.config.noise_dim * 2) # (*2) for mean and variance

        self.apply(weights_init)

    def forward(self, patch):

        out = self.conv1(patch)
        print("************** ", out.shape)
        out = self.relu(self.batch_norm1(out))

        out = self.conv2(out)
        print("************** ", out.shape)
        out = self.relu(self.batch_norm2(out))

        out = self.conv3(out)
        print("************** ", out.shape)
        out = self.relu(self.batch_norm3(out))

        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3]*out.shape[4])
        out = self.linear(out)

        return torch.chunk(out, 2, 1)


"""
netG testing
"""
def main():
    config = json.load(open('../../configs/gan_exp_0.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.noise_dim))
    print (inp.shape)
    netD = Generator(config)
    out = netD(inp)
    print (out.shape)

if __name__ == '__main__':
    main()
