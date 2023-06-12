import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from modules import *

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, int(self.size) * int(self.size)).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, int(self.size), int(self.size))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x=None):
        x = self.up(x)
        if skip_x is not None:
            x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x

class Swin_Unet(nn.Module):
    def __init__(self, net_dimension=256, img_size=224, c_out=3) -> None:
        super().__init__()

        # Define the pretrained Encoder
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        self.model = model.features

        # Define the bottleneck
        self.bot1 = DoubleConv(768, 512)
        self.bot2 = DoubleConv(512, 768)

        # Define the decoder
        self.up1 = Up(384+(768*2), net_dimension*4)
        self.sa1 = SelfAttention(net_dimension*4, img_size//16)

        self.up2 = Up(192+net_dimension*4, net_dimension*2)
        self.sa2 = SelfAttention(net_dimension*2, img_size//8)

        self.up3 = Up(96+net_dimension*2, net_dimension)
        self.sa3 = SelfAttention(net_dimension, img_size//4)

        self.up4 = Up(net_dimension, net_dimension//4)
        self.sa4 = SelfAttention(net_dimension//4, img_size//2)

        self.up5 = Up(net_dimension//4, net_dimension//8)

        self.outc = nn.Sequential(
            nn.Conv2d(net_dimension//8, c_out, kernel_size=1),
        )

    def forward(self, x, labels):
        # List of features
        list_features = []
        
        # Loop to get the features of sub layers and final output
        for layer in self.model:
            x = layer(x)
            list_features.append(x.swapaxes(1,3))

        x = x.swapaxes(1,3)

        # Bottleneck
        x = self.bot1(x)
        x = self.bot2(x)
        x = torch.cat([x, labels.view(-1,768,7,7)], dim=1)

        # Decoder
        x = self.up1(x, list_features[4])
        x = self.sa1(x)

        x = self.up2(x, list_features[2])
        x = self.sa2(x)

        x = self.up3(x, list_features[0])
        x = self.sa3(x)

        x = self.up4(x)
        # x = self.sa4(x)
        x = self.up5(x)

        x = self.outc(x)

        return x

if __name__ == "__main__":
    print("Main")
    
    # Images create
    batch_size = 2
    img = torch.rand((batch_size, 3, 224, 224)).to("cuda")
    label = torch.rand((batch_size, 49, 768)).to("cuda")
    # Forward
    model = Swin_Unet().to("cuda")
    # model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).to("cuda")
    out = model(img, label)
    print(out.shape)
    # print(out[0].shape)

    # for i in out[1]:
    #     print(i.shape)