import torch
import torch.nn as nn
import torch.nn.functional as F
# vision transform
from ViT import *

# https://github.com/milesial/Pytorch-UNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ColorNetwork(nn.Module):
    def __init__(self, n_channels, n_classes, img_size=128, ch_deep=32, bilinear=False):
        super(ColorNetwork, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ch_deep = ch_deep

        self.inc = (DoubleConv(n_channels, self.ch_deep))
        self.down1 = (Down(self.ch_deep, self.ch_deep*2))
        self.down2 = (Down(self.ch_deep*2, self.ch_deep*4))
        self.down3 = (Down(self.ch_deep*4, self.ch_deep*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.ch_deep*8, (self.ch_deep*16) // factor))

        self.vit = Vit_neck(64, self.ch_deep*8, (self.ch_deep*16)*(img_size//16)**2)
        
        self.up1 = (Up(self.ch_deep*32, self.ch_deep*8 // factor, bilinear))
        self.up2 = (Up(self.ch_deep*8, self.ch_deep*4 // factor, bilinear))
        self.up3 = (Up(self.ch_deep*4, self.ch_deep*2 // factor, bilinear))
        self.up4 = (Up(self.ch_deep*2, self.ch_deep, bilinear))
        self.outc = (OutConv(self.ch_deep, n_classes))

    def forward(self, x, color_sample):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # print(f"x4 shape: {x4.shape}")

        x5 = self.down4(x4)
        # print(f"x5 shape: {x5.shape}")

        neck = self.vit(color_sample)
        # print(f"neck v1 shape: {neck.shape}")
        
        neck = torch.reshape(neck, (x5.shape[0], x5.shape[1], x5.shape[2], x5.shape[3]))
        # print(f"neck v2 shape: {neck.shape}")

        x5 = torch.cat((neck, x5), 1)
        # print(f"x5 v2 shape: {x5.shape}")

        x = self.up1(x5, neck)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)

# model = ColorNetwork(1,3).to("cuda")
# x = torch.zeros((10,1,128,128)).to("cuda")
# color_sample = torch.ones((10,3,128,128)).to("cuda")


# model(x, color_sample).shape