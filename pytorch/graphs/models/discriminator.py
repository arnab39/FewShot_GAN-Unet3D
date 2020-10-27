import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from easydict import EasyDict as edict
from graphs.models.custom_functions.weight_norm import WN_Conv3d, WN_ConvTranspose3d

# 3D-UNet as Discriminator
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.input_channel = self.config.num_modalities
        self.num_classes = self.config.num_classes
        kernel_size = (3,3,3)
        kernel_size_deconv = (2,2,2)
        stride_deconv = (2,2,2)
        out_channels = 32
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout3d(p=0.2, inplace=False)
        self.final_activation = nn.Softmax(dim=1)

        # Defining the convolutional operations
        self.pool = nn.AvgPool3d(2)

        self.encoder0 = WN_Conv3d(self.input_channel, out_channels, kernel_size)
        self.encoder1 = WN_Conv3d(out_channels, out_channels, kernel_size)

        self.encoder2 = WN_Conv3d(out_channels, out_channels*(2), kernel_size)
        self.encoder3 = WN_Conv3d(out_channels*(2), out_channels*(2), kernel_size)

        self.encoder4 = WN_Conv3d(out_channels*(2), out_channels*(2**2), kernel_size)
        self.encoder5 = WN_Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size)

        self.encoder6 = WN_Conv3d(out_channels*(2**2), out_channels*(2**3), kernel_size)
        self.encoder7 = WN_Conv3d(out_channels*(2**3), out_channels*(2**3), kernel_size)

        self.decoder1 = WN_ConvTranspose3d(out_channels*(2**3), out_channels*(2**3), kernel_size_deconv, stride_deconv)
        #encoder5 + decoder1
        self.encoder8 = WN_Conv3d(out_channels*(2**2) + out_channels*(2**3), out_channels*(2**2), kernel_size)
        self.encoder9 = WN_Conv3d(out_channels*(2**2), out_channels*(2**2), kernel_size)

        self.decoder2 = WN_ConvTranspose3d(out_channels*(2**2), out_channels*(2**2), kernel_size_deconv, stride_deconv)
        #encoder3 + decoder2
        self.encoder10 = WN_Conv3d(out_channels*(2) + out_channels*(2**2), out_channels*(2), kernel_size)
        self.encoder11 = WN_Conv3d(out_channels*(2), out_channels*(2), kernel_size)

        self.decoder3 = WN_ConvTranspose3d(out_channels*(2), out_channels*(2), kernel_size_deconv, stride_deconv)
        #encoder1 + decoder3
        self.encoder12 = WN_Conv3d(out_channels + out_channels*(2), out_channels, kernel_size)
        self.encoder13 = WN_Conv3d(out_channels, out_channels, kernel_size)

        self.final_conv = WN_Conv3d(out_channels, self.num_classes, kernel_size)

    def forward(self, input, get_feature=False, use_dropout=False):
        conv0 = self.lrelu(self.encoder0(input))
        conv1 = self.lrelu(self.encoder1(conv0))
        pool1 = self.pool(conv1)

        conv2 = self.lrelu(self.encoder2(pool1))
        conv3 = self.lrelu(self.encoder3(conv2))
        pool3 = self.pool(conv3)

        conv4 = self.lrelu(self.encoder4(pool3))
        conv5 = self.lrelu(self.encoder5(conv4))
        pool5 = self.pool(conv5)

        conv6 = self.lrelu(self.encoder6(pool5))
        conv7 = self.lrelu(self.encoder7(conv6))

        if use_dropout:
            conv7 = self.dropout(conv7)

        deconv1 = self.decoder1(conv7)
        skip_connection1 = torch.cat((conv5, deconv1), 1)
        conv8 = self.lrelu(self.encoder8(skip_connection1))
        conv9 = self.lrelu(self.encoder9(conv8))

        deconv2 = self.decoder2(conv9)
        skip_connection2 = torch.cat((conv3, deconv2), 1)
        conv10 = self.lrelu(self.encoder10(skip_connection2))
        conv11 = self.lrelu(self.encoder11(conv10))

        deconv3 = self.decoder3(conv11)
        skip_connection3 = torch.cat((conv1, deconv3), 1)
        conv12 = self.lrelu(self.encoder12(skip_connection3))
        conv13 = self.lrelu(self.encoder13(conv12))

        if use_dropout:
            conv13 = self.dropout(conv13)

        final_output = self.final_conv(conv13)

        if not get_feature:
            return final_output, self.final_activation(final_output)
        else:
            return final_output, self.final_activation(final_output), conv6


"""
netD testing
"""
def main():
    config = json.load(open('../../configs/gan_exp_0.json'))
    config = edict(config)
    inp  = torch.autograd.Variable(torch.randn(config.batch_size, config.input_channels, config.patch_shape[0], config.patch_shape[1], config.patch_shape[2]))
    print (inp.shape)
    netD = Discriminator(config)
    out = netD(inp)
    print (out)

if __name__ == '__main__':
    main()
