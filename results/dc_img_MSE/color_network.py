
__author__ = 'lstival'

# https://github.com/L1aoXingyu
# https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
# https://huggingface.co/models?sort=downloads

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ViT import *

# Loss imports
from piq import ssim, SSIMLoss

# Images of training epochs
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

# Models Save
if not os.path.exists('./models'):
    os.mkdir('./models')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x


# batch_size = 128

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# dataset = MNIST('./data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import load_data as ld
dataLoader = ld.ReadData()
import torch

# Spatial size of training images. All images will be resized to this
image_size = (256, 256)

# Batch size during training
batch_size = 32
num_epochs = 100

# Root directory for dataset
dataroot = "../data/train/sunset"

dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size)

def ConvDown2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1):

    conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
    return conv

def ConvUp2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1):

    conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding)
    return conv

class ColorNetwork(nn.Module):

    def __init__(self, in_channel, out_channel, stride, padding) -> None:
        super(ColorNetwork, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = 3
        self.stride = stride
        self.padding = padding

        #Encoder Network
        self.dw_conv1 = ConvDown2d(1, self.out_channel, self.kernel_size,self.stride,self.padding)
        self.max_pol1 = nn.MaxPool2d(2, stride=1)

        self.dw_conv2 = ConvDown2d(self.out_channel, self.out_channel*2, self.kernel_size,self.stride,self.padding)
        self.max_pol2 = nn.MaxPool2d(2, stride=1)

        self.dw_conv3 = ConvDown2d(self.out_channel*2, self.out_channel*4, self.kernel_size,self.stride,self.padding)
        self.max_pol3 = nn.MaxPool2d(2, stride=1)

        self.dw_conv4 = ConvDown2d(self.out_channel*4, self.out_channel*4, self.kernel_size,self.stride,self.padding)
        self.max_pol4 = nn.MaxPool2d(2, stride=1)

        #Decoder
        self.up_conv4 = ConvUp2d(self.out_channel*8, self.out_channel*4, 2, self.stride, 0)

        self.up_conv3 = ConvUp2d(self.out_channel*8, self.out_channel*2, 2, self.stride, 0)

        self.up_conv2 = ConvUp2d(self.out_channel*4, self.out_channel, 2, self.stride,0)

        self.up_conv1 = ConvUp2d(self.out_channel*2, 3, 2, 2, 0)

        #Activation
        self.activation = nn.Tanh()

    def forward(self, x, color_sample) -> torch.Tensor:

        #Encoder
        e1 = self.dw_conv1(x)
        e1 = nn.Tanh()(e1)
        e1 = self.max_pol1(e1)
        # print(f"e1 shape: {e1.shape}")

        e2 = self.dw_conv2(e1)
        e2 = nn.Tanh()(e2)
        e2 = self.max_pol2(e2)
        # print(f"e2 shape: {e2.shape}")

        e3 = self.dw_conv3(e2)
        e3 = nn.Tanh()(e3)
        e3 = self.max_pol3(e3)
        # print(f"e3 shape: {e3.shape}")

        e4 = self.dw_conv4(e3)
        e4 = nn.Tanh()(e4)
        e4 = self.max_pol4(e4)
        # print(f"e4 shape: {e4.shape}")

        #BottlerNeck
        neck = vit.forward(color_sample)
        # print(f"neck shape: {neck.shape}")
        neck = torch.reshape(neck, (e4.shape[0], e4.shape[1], e4.shape[2], e4.shape[3]))
        e4 = torch.cat((neck, e4), 1)
        # print(f"e4 shape: {e4.shape}")

        #Decoder
        d4 = self.up_conv4(e4)
        d4 = nn.Tanh()(d4)
        # print(f"d4 shape: {d4.shape}")
        
        d3 = torch.cat((e3, d4), 1)
        # print(f"d3 shape: {d3.shape}")
        d3 = self.up_conv3(d3)
        d3 = nn.Tanh()(d3)

        d2 = torch.cat((e2, d3), 1)
        d2 = self.up_conv2(d2)
        d2 = nn.Tanh()(d2)

        d1 = torch.cat((e1, d2), 1)
        d1 = self.up_conv1(d1)
        d1 = nn.Tanh()(d1)

        #Activation
        out = self.activation(d1)
        return out

## Training
model = ColorNetwork(in_channel=1, out_channel=32, stride=2, padding=2).cuda()

#Visual Transformer
conv_dim = image_size[0] / 16
vit = Vit_neck(batch_size, image_size[0], int(128*conv_dim*conv_dim))

# https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/#:~:text=awesome%20concept%20now.-,What%20are%20Skip%20Connections%3F,different%20problems%20in%20different%20architectures.
learning_rate = 1e-2

# ================ Losses ========================
LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
# SSIM = SSIMLoss(data_range=1.)
MSE = nn.MSELoss()

criterion = MSE
criterion.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=1e-5)

for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        img, _ = data
        img.cuda()
        # img_gray  = img[:,:1,:,:].cuda()
        img_gray = transforms.Grayscale(num_output_channels=1)(img).cuda()
        img_gray = Variable(img_gray).cuda()
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img_gray, img.to("cuda"))
        # loss = criterion(to_img(output), to_img(img.cuda()))
        loss = criterion(output.to("cuda"), img.to("cuda"))
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, total_loss))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
        torch.save(model.state_dict(), f'./models/color_netowrk_end{epoch}.pth')

torch.save(model.state_dict(), './models/color_netowrk_end.pth')