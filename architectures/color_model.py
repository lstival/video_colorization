# vision transform
from architectures.ViT import *

from torch import nn
import torch

def ConvDown2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1):
    """
    Function with the convolutional layer to decrease "encoder" the data.
    """

    conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
    return conv

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
        # self.vit = vit

        self.img_size = img_size
        # Factor of the output of Vit
        conv_dim = img_size / 16

        #Encoder Network
        self.dw_conv1 = ConvDown2d(1, self.out_channel, self.kernel_size,self.stride,self.padding)
        self.max_pol1 = nn.MaxPool2d(2, stride=1)

        self.dw_conv2 = ConvDown2d(self.out_channel, self.out_channel*2, self.kernel_size,self.stride,self.padding)
        self.max_pol2 = nn.MaxPool2d(2, stride=1)

        self.dw_conv3 = ConvDown2d(self.out_channel*2, self.out_channel*4, self.kernel_size,self.stride,self.padding)
        self.max_pol3 = nn.MaxPool2d(2, stride=1)

        self.dw_conv4 = ConvDown2d(self.out_channel*4, self.out_channel*4, self.kernel_size,self.stride,self.padding)
        self.max_pol4 = nn.MaxPool2d(2, stride=1)

        #Visual Transformer definiton (Batchsize, Image input size, size of tensor to be reshaped to "encaixar" in the last encoder layer)
        self.vit = Vit_neck(64, self.img_size, int(int(self.out_channel*4)*conv_dim*conv_dim))

        #Upscaling
        self.up_conv4 = ConvUp2d(self.out_channel*8, self.out_channel*4, 2, self.stride, 0)
        # self.bat4 = nn.BatchNorm2d(self.out_channel*4)

        self.up_conv3 = ConvUp2d(self.out_channel*8, self.out_channel*2, 2, self.stride, 0)
        # self.bat3 = nn.BatchNorm2d(self.out_channel*2)

        self.up_conv2 = ConvUp2d(self.out_channel*4, self.out_channel, 2, self.stride,0)
        # self.bat2 = nn.BatchNorm2d(self.out_channel)

        self.up_conv1 = ConvUp2d(self.out_channel*2, 3, 2, 2, 0)

        #Activation
        self.activation = nn.Tanh()

    def forward(self, x, color_sample) -> torch.Tensor:

        #Encoder
        e1 = self.dw_conv1(x)
        e1 = nn.Tanh()(e1)
        e1 = self.max_pol1(e1)
        
        e2 = self.dw_conv2(e1)
        e2 = nn.Tanh()(e2)
        e2 = self.max_pol2(e2)

        e3 = self.dw_conv3(e2)
        e3 = nn.Tanh()(e3)
        e3 = self.max_pol3(e3)

        e4 = self.dw_conv4(e3)
        e4 = nn.Tanh()(e4)
        e4 = self.max_pol4(e4)
        # print(f"e4_v1 max: {e4.max()}")
        # print(f"e4 shape: {e4.shape}")

        #BottlerNeck
        neck =  self.vit(color_sample)
        # print(f"neck shape: {neck.shape}")
        # neck = torch.reshape(neck, (1, e4.shape[1], e4.shape[2], e4.shape[3]))
        neck = torch.reshape(neck, (e4.shape[0], e4.shape[1], e4.shape[2], e4.shape[3]))
        e4 = torch.cat((neck, e4), 1)
        # print(f"e4_v1 max: {e4.max()}")

        #Decoder
        d4 = self.up_conv4(e4)
        d4 = nn.Tanh()(d4)
        # d4 = self.bat4(d4)
        
        d3 = torch.cat((e3, d4), 1)
        d3 = self.up_conv3(d3)
        d3 = nn.Tanh()(d3)
        # d3 = self.bat4(d3)

        d2 = torch.cat((e2, d3), 1)
        d2 = self.up_conv2(d2)
        d2 = nn.Tanh()(d2)
        # d2 = self.bat4(d2)

        d1 = torch.cat((e1, d2), 1)
        d1 = self.up_conv1(d1)
        # d1 = nn.Tanh()(d1)

        #Activation
        out = self.activation(d1)
        return out