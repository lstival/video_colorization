
from torchvision import models
from torch import nn
import torch

# https://pytorch.org/vision/master/models/generated/torchvision.models.vgg16.html

vgg = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
# vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)


class VGG_encoder(nn.Module):

    def __init__(self) -> None:
        super(VGG_encoder, self).__init__()
        # self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # self.vgg_encoder = nn.Sequential(self.vgg.features)
        # self.vgg_encoder = nn.Sequential(*list(vgg.features.children()))

        # encoding
        self.conv_0 = nn.Sequential(*list(vgg.features.children())[:3])
        self.conv_1 = nn.Sequential(*list(vgg.features.children())[3:6])
        self.conv_2 = nn.Sequential(*list(vgg.features.children())[6:8])
        self.conv_3 = nn.Sequential(*list(vgg.features.children())[8:10])
        self.conv_4 = nn.Sequential(*list(vgg.features.children())[10:])

        # upscaling
        self.up_conv4 = nn.ConvTranspose2d(256*2,256,3,2)
        self.up_conv3 = nn.ConvTranspose2d(256*2,384,3,1,1)
        self.up_conv2 = nn.ConvTranspose2d(384*2,192,3,1,1)
        self.up_conv1 = nn.ConvTranspose2d(192*2,64,3,2)
        self.up_conv0 = nn.ConvTranspose2d(64*2,3,11,4,2)

    def forward(self, images):
        
        # encoder
        x0 = self.conv_0(images)
        # print(f"conv_0 shape {x0.shape}")
        x1 = self.conv_1(x0)
        # print(f"conv_1 shape {x1.shape}")
        x2 = self.conv_2(x1)
        # print(f"conv_2 shape {x2.shape}")
        x3 = self.conv_3(x2)
        # print(f"conv_3 shape {x3.shape}")
        x4 = self.conv_4(x3)
        # print(f"conv_4 shape {x4.shape}")

        # upscale
        # neck = torch.cat((x4,x4),1)
        # print(f"neck shape {neck.shape}")

        # d4 = self.up_conv4(neck)
        # print(f"Up_conv_4 shape {d4.shape}")

        # d4 = torch.cat((x3,d4), 1)
        # d3 = self.up_conv3(d4)
        # print(f"Up_conv_3 shape {d3.shape}")

        # d3 = torch.cat((x2,d3), 1)
        # d2 = self.up_conv2(d3)
        # print(f"Up_conv_2 shape {d2.shape}")

        # d2 = torch.cat((x1,d2), 1)
        # d1 = self.up_conv1(d2)
        # print(f"Up_conv_1 shape {d1.shape}")

        # d1 = torch.cat((x0,d1), 1)
        # d0 = self.up_conv0(d1)
        # print(f"Up_conv_1 shape {d0.shape}")

        # out = d0
        return x4,x3,x2,x1,x0
        # return out
    


# aa = torch.zeros(32,3,256,256)
# model = VGG_encoder()
# out = model(aa)
# print(f" Same shape {aa.shape == out.shape}")
# Get intermediate layers features (outputs)
# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html