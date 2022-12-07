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

    def __init__(self, in_channel, out_channel, stride, padding, img_size) -> None:
        super(ColorNetwork, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 3
        self.stride = stride
        self.padding = padding
        self.img_size = img_size

        # Factor of the output of Vit
        # self.vit = Vit_neck(32, 256, int(256*7*7))
        self.vit = Vit_neck(32, self.img_size, int(512*7*7))
        
        #Encoder Network

        self.vgg_encoder = VGG_encoder()

        # Upscaling
        self.up_conv4 = nn.ConvTranspose2d(256*2,256,3,2)
        self.up_conv3 = nn.ConvTranspose2d(256*2,384,3,1,1)
        self.up_conv2 = nn.ConvTranspose2d(384*2,192,3,1,1)
        self.up_conv1 = nn.ConvTranspose2d(192*2,64,3,2)
        self.up_conv0 = nn.ConvTranspose2d(64*2,3,11,4,2)
        self.up_conv = nn.ConvTranspose2d(3,3,4,2)
        # self.up_conv4 = ConvUp2d(256*2, self.out_channel*8, 2, 1, 0)

        # self.up_conv3 = ConvUp2d(self.out_channel*8, self.out_channel*4, 2, self.stride, 0)

        # self.up_conv2 = ConvUp2d(self.out_channel*4, self.out_channel*2, 2, self.stride,0)

        # self.up_conv1 = ConvUp2d(self.out_channel*2, self.out_channel, 2, 2, 0)

        # self.up_conv0 = ConvUp2d(self.out_channel, out_channel, 2, 2, 0)

        # self.up_conv = ConvUp2d(256, 3, 2, 2, 0)

        #Activation
        self.activation = nn.Tanh()

        print("Using Alex Net")

    def forward(self, x, color_sample) -> torch.Tensor:

        #Encoder
        # Create a three channels in gray image
        # x = torch.cat((x, x, x), 1)
        x4,x3,x2,x1,x0 =  self.vgg_encoder(x)
        # print(f"x4_v1 shape: {x4.shape}")
        # print(f"x4_v1 max: {x4.max()}")

        #BottlerNeck
        # neck =  self.vgg_encoder(color_sample)
        neck = self.vit.forward(color_sample)
        neck = torch.reshape(neck, (x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3]))
        # print(f"neck shape: {neck.shape}")
        # neck = torch.reshape(neck, (x4.shape[0], x4.shape[1], x4.shape[2], x4.shape[3]))
        x4 = torch.cat((neck, x4), 1)
        # print(f"x4 max: {x4.max()}")
        # print(f"x4_v2 shape: {x4.shape}")

        #Decoder
        d4 = self.up_conv4(x4)
        d4 = nn.Tanh()(d4)
        # print(f"Up_conv_4 shape {d4.shape}")

        d4 = torch.cat((x3,d4), 1)
        d3 = self.up_conv3(d4)
        d4 = nn.Tanh()(d3)
        # print(f"Up_conv_3 shape {d3.shape}")

        d3 = torch.cat((x2,d3), 1)
        d2 = self.up_conv2(d3)
        d4 = nn.Tanh()(d2)
        # print(f"Up_conv_2 shape {d2.shape}")

        d2 = torch.cat((x1,d2), 1)
        d1 = self.up_conv1(d2)
        d4 = nn.Tanh()(d1)
        # print(f"Up_conv_1 shape {d1.shape}")

        d1 = torch.cat((x0,d1), 1)
        d0 = self.up_conv0(d1)
        d0 = nn.Tanh()(d0)

        d = self.up_conv(d0)
        # print(f"Up_conv_1 shape {d0.shape}")
        
        # d4 = self.up_conv4(e4)
        # d4 = nn.Tanh()(d4)
        # # print(f"d4 shape: {d4.shape}")
        
        # # d3 = torch.cat((e3, d4), 1)
        # # d3 = self.up_conv3(d3)
        # d3 = self.up_conv3(d4)
        # d3 = nn.Tanh()(d3)
        # # print(f"d3 shape: {d3.shape}")

        # # d2 = torch.cat((e2, d3), 1)
        # # d2 = self.up_conv2(d2)
        # d2 = self.up_conv2(d3)
        # d2 = nn.Tanh()(d2)
        # # print(f"d2 shape: {d2.shape}")

        # # d1 = torch.cat((e1, d2), 1)
        # # d1 = self.up_conv1(d1)
        # d1 = self.up_conv1(d2)
        # d1 = nn.Tanh()(d1)
        # # print(f"d1 shape: {d1.shape}")

        # d0 = self.up_conv0(d1)
        # d0 = nn.Tanh()(d0)
        # # print(f"d0 shape: {d0.shape}")

        # d = self.up_conv(d0)
        # d = nn.Tanh()(d)
        # # print(f"d max: {d.max()}")

        #Activation
        out = self.activation(d)
        # print(f"out max: {out.max()}")
        return out
    
# model = ColorNetwork(in_channel=1, out_channel=64, stride=2, padding=2, img_size=256).cuda()
# sample = torch.ones(16,3,256,256)
# out = model(sample.cuda(), sample.cuda())