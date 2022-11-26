# vision transform
from ViT import *

from torch import nn
import torch
from vgg_encoder import VGG_encoder

def ConvUp2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1):
    """
    Function to upscale back the tensor to original iamge size.
    """
    conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
    return conv

class ColorNetwork(nn.Module):
    """
    Network to colorization, compsed by an group of ConvDown2d to learn how
    create feature space of the gray scale iamge.
    Other group ConvUp2d to resize the image to the original size.
    The botlerneck is a Vit (Vision Transform) responsable to create a feature
    space of the sample image (with the wished colors)
    All activations are presented by Tanh() function.

    Return a Nx3xHxW image with colors.
    """

    def __init__(self, in_channel, out_channel, stride, padding) -> None:
        super(ColorNetwork, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 3
        self.stride = stride
        self.padding = padding

        # Factor of the output of Vit
        self.vit = Vit_neck(32, 256, int(256*7*7))
        
        #Encoder Network

        self.vgg_encoder = VGG_encoder()

        #Decoder
        self.up_conv4 = ConvUp2d(256*2, self.out_channel*8, 2, 1, 0)

        self.up_conv3 = ConvUp2d(self.out_channel*8, self.out_channel*4, 2, self.stride, 0)

        self.up_conv2 = ConvUp2d(self.out_channel*4, self.out_channel*2, 2, self.stride,0)

        self.up_conv1 = ConvUp2d(self.out_channel*2, self.out_channel, 2, 2, 0)

        self.up_conv0 = ConvUp2d(self.out_channel, out_channel, 2, 2, 0)

        self.up_conv = ConvUp2d(self.out_channel, 3, 2, 2, 0)

        #Activation
        self.activation = nn.Tanh()

        print("Using Alex Net")

    def forward(self, x, color_sample) -> torch.Tensor:

        #Encoder
        # Create a three channels in gray image
        # x = torch.cat((x, x, x), 1)
        e4 =  self.vgg_encoder(x)
        # print(f"e4_v1 shape: {e4.shape}")
        # print(f"e4_v1 max: {e4.max()}")

        #BottlerNeck
        # neck =  self.vgg_encoder(color_sample)
        neck = self.vit.forward(color_sample)
        neck = torch.reshape(neck, (e4.shape[0], e4.shape[1], e4.shape[2], e4.shape[3]))
        # print(f"neck shape: {neck.shape}")
        # neck = torch.reshape(neck, (e4.shape[0], e4.shape[1], e4.shape[2], e4.shape[3]))
        e4 = torch.cat((neck, e4), 1)
        # print(f"e4 max: {e4.max()}")
        # print(f"e4_v2 shape: {e4.shape}")

        #Decoder
        d4 = self.up_conv4(e4)
        d4 = nn.Tanh()(d4)
        # print(f"d4 shape: {d4.shape}")
        
        # d3 = torch.cat((e3, d4), 1)
        # d3 = self.up_conv3(d3)
        d3 = self.up_conv3(d4)
        d3 = nn.Tanh()(d3)
        # print(f"d3 shape: {d3.shape}")

        # d2 = torch.cat((e2, d3), 1)
        # d2 = self.up_conv2(d2)
        d2 = self.up_conv2(d3)
        d2 = nn.Tanh()(d2)
        # print(f"d2 shape: {d2.shape}")

        # d1 = torch.cat((e1, d2), 1)
        # d1 = self.up_conv1(d1)
        d1 = self.up_conv1(d2)
        d1 = nn.Tanh()(d1)
        # print(f"d1 shape: {d1.shape}")

        d0 = self.up_conv0(d1)
        d0 = nn.Tanh()(d0)
        # print(f"d0 shape: {d0.shape}")

        d = self.up_conv(d0)
        d = nn.Tanh()(d)
        # print(f"d max: {d.max()}")

        #Activation
        out = self.activation(d)
        # print(f"out max: {out.max()}")
        return out
    
# model = ColorNetwork(in_channel=1, out_channel=64, stride=2, padding=2).cuda()
# sample = torch.ones(16,1,256,256)
# out = model(sample.cuda(), torch.cat((sample, sample, sample), 1).cuda())