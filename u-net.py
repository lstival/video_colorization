__author__ = 'SherlockLiao'

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
    x = x.view(x.size(0), 3, 256, 256)
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


class ConvDown2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1) -> None:
        super(ConvDown2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x) -> torch.Tensor:
        conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, self.stride, self.padding),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        return conv(x)

class ConvUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1) -> None:
        super(ConvUp2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        conv = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel, self.out_channel, self.kernel_size, self.stride, self.padding),
            nn.ReLU(True),
        )
        return conv(x)

class ColorNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super(ColorNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        #Encoder Network
        self.encoder_1 = ConvDown2d(self.in_channels, self.out_channels, self.stride)
        self.encoder_2 = ConvDown2d(self.out_channels, self.out_channels*2, self.stride)
        self.encoder_3 = ConvDown2d(self.out_channels*2, self.out_channels*4, self.stride)

        #Decoder
        self.decoder_1 = ConvUp2d(self.out_channels*4, self.out_channels*2, self.stride)
        self.decoder_2 = ConvUp2d(self.out_channels*2, self.out_channels, self.stride)
        self.decoder_3 = ConvUp2d(self.out_channels, 3, self.stride)

        #Activation
        self.activation = nn.Tanh()

    def forward(self, x) -> torch.Tensor:

        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)

        d1 = self.decoder_1(e3)
        d2 = self.decoder_2(d1)
        d3 = self.decoder_3(d2)

        out = self.activation(d3)
        return out

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x, img_color):
        x = self.encoder(x)
        # print(f"Pre Vit: {x.shape}")
        x1 = vit.forward(img_color)
        # print(f"Pre reshape {x1.shape}")
        x1 = torch.reshape(x1, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        # print(f"Reshapado {x1.shape}")
        x2 = torch.add(x.to("cuda"), x1.to("cuda"))
        # print(f"Pos add {x2.shape}")

        x = self.decoder(x2)
        return x


model = autoencoder().cuda()

#Visual Transformer
vit = Vit_neck(batch_size, 256, 8*21*21)

criterion = nn.MSELoss()
learning_rate = 1e-3

# criterion = SSIMLoss(data_range=1.)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        img, _ = data
        img.cuda()
        img_gray  = img[:,:1,:,:].cuda()
        img_gray = Variable(img_gray).cuda()
        # ===================forward=====================
        output = model(img_gray, img)
        # loss = criterion(to_img(output), to_img(img.cuda()))
        loss = criterion(output, img.to("cuda"))
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
        torch.save(model.state_dict(), f'./models/conv_autoencoder_{epoch}.pth')

torch.save(model.state_dict(), './models/conv_autoencoder_end.pth')