import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from modules import *
# Import the optical flow network

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Swin_Unet(nn.Module):
    def __init__(self, net_dimension=256, img_size=224, c_out=3) -> None:
        super().__init__()

        # Define the bottleneck
        self.bot1 = DoubleConv(768, 256)

        # Color Agreement
        self.color1 = DoubleConv(512, 256)
        self.color2 = DoubleConv(256, 128)
        self.color3 = DoubleConv(128, 64)

        # Flow Agreement
        self.flow1 = Down(2, 16)
        self.flow2 = Down(16, 32)
        self.flow3 = Down(32, 64)
        self.flow4 = Down(64, 128)
        self.flow5 = Down(128, 256)

        # Define the decoder
        self.up1 = Up((384+196+256+256), net_dimension*4)
        self.sa1 = SelfAttention(net_dimension*4, img_size//16)

        self.up2 = Up(192+(net_dimension*4), net_dimension*2)
        self.sa2 = SelfAttention(net_dimension*2, img_size//8)

        self.up3 = Up(96+(net_dimension*2), net_dimension)
        self.sa3 = SelfAttention(net_dimension, img_size//4)

        self.up4 = Up(net_dimension, net_dimension//4)
        self.sa4 = SelfAttention(net_dimension//4, img_size//2)

        self.up5 = Up(1+(net_dimension//4), net_dimension//8)

        self.outc = nn.Sequential(
            nn.Conv2d(net_dimension//8, c_out, kernel_size=1),
        )

    def forward(self, x, labels, flow, swin_model):
        # List of features
        list_features = []
        
        # Loop to get the features of sub layers and final output
        x_inpt = x
        for layer in swin_model:
            x = layer(x)
            list_features.append(x.swapaxes(1,3))

        x = x.swapaxes(1,3)

        # Bottleneck
        x = self.bot1(x)
        # x = self.bot2(x)
        # x = torch.cat([x, labels.view(-1, 1568, 4, 4)], dim=1)
        x_color = self.color1(labels)
        x_color = self.color2(x_color)
        x_color = self.color3(x_color)
        x_color = x_color.view(-1, 196, 4, 4)

        # Flow
        x_flow = self.flow1(flow)
        x_flow = self.flow2(x_flow)
        x_flow = self.flow3(x_flow)
        x_flow = self.flow4(x_flow)
        x_flow = self.flow5(x_flow)
        
        x = torch.cat([x, x_color, x_flow], dim=1)

        # Decoder
        x = self.up1(x, list_features[5])
        x = self.sa1(x)

        x = self.up2(x, list_features[3])
        x = self.sa2(x)

        x = self.up3(x, list_features[1])
        x = self.sa3(x)

        x = self.up4(x)
        # x = self.sa4(x)
        x = self.up5(x, x_inpt[:,:1])

        x = self.outc(x)

        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    from flow import Flow_Network
    
    print("Main")
    
    # Images create
    batch_size = 2
    img_size = 128
    net_dimension=128
    img = torch.rand((batch_size, 3, img_size, img_size)).to("cuda")
    # label = torch.rand((batch_size, 49, 768)).to("cuda")
    label = torch.rand((batch_size, 512, 7, 7)).to("cuda")
    
    # Define the pretrained Encoder
    swin_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).to("cuda").features

    # Define the pretrained Flow Network
    flow_model = Flow_Network().to("cuda")

    # Forward
    model = Swin_Unet(net_dimension=net_dimension, img_size=img_size).to("cuda")
    flow_out = flow_model(img, img)
    # model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).to("cuda")
    out = model(img, label, flow_out, swin_model)
    print(out.shape)
    print(f"Network parameters: {count_parameters(model)/1000000}")

    # print(flow_out.shape)

    # print(out[0].shape)

    # for i in out[1]:
    #     print(i.shape)