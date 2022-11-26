__author__ = 'lstival'

# https://github.com/L1aoXingyu
# https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
# https://huggingface.co/models?sort=downloads

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
# import bottleneck
# import pandas as pd

#  dataloader class
import load_data as ld

# vision transform
from ViT import *

from edge_detection import *

# random seed to torch 
torch.manual_seed(2021)

# Images of training epochs
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

# Models Save
if not os.path.exists('./models'):
    os.mkdir('./models')

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

# ================ Read Data =====================
dataLoader = ld.ReadData()

# Root directory for dataset
# dataroot = "../data/train/woman"

dataroot = "C:/video_colorization/data/train/diverses"

# Spatial size of training images. All images will be resized to this
image_size = (256, 256)

# Batch size during training
batch_size = 16
num_epochs = 101

dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size)

# from color_model_vgg import ColorNetwork
from color_model import ColorNetwork

# Factor of the output of Vit
conv_dim = image_size[0] / 16

#Visual Transformer definiton (Batchsize, Image input size, size of tensor to be reshaped to "encaixar" in the last encoder layer)
# vit = Vit_neck(batch_size, image_size[0], int(256*7*7))

## Model setings
model = ColorNetwork(in_channel=1, out_channel=64, stride=2, padding=2).cuda()

# ================ Losses ========================

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# from torchmetrics import PeakSignalNoiseRatio
# from torchmetrics.image.fid import FrechetInceptionDistance
from piq import ssim, SSIMLoss
# from vgg_loss import VGGLoss, TVLoss

# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='alex')
# PSNR = PeakSignalNoiseRatio()
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIMLoss(data_range=1.)

# Falta Vram
# PERCEPTUAL = VGGLoss(model='vgg19')
# FID = FrechetInceptionDistance(feature=64)
# LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

# import lpips
# LPIPS = lpips.LPIPS(net='alex')

# ================ Train ======================== 

# learning rate of the netowrk
learning_rate = 1e-2

dic_losses = {}

# define the loss
criterion = MSE
criterion.cuda()

# selection of the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=1e-5)

# ed = EdgeExctration(64, 256)

# loop to train
for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        img, _ = data
        img.cuda()
        img_gray = transforms.Grayscale(num_output_channels=1)(img).cuda()

        # img_gray = ed.get_edges(img, 30, 150)
        # print(f"img_gray shape1: {len(img_gray)}")
        # img_gray = torch.stack(img_gray, 0).unsqueeze(1)
        # print(f"img_gray shape2: {img_gray.shape}")
        # img_gray.cuda()
    
        img_gray = Variable(img_gray).cuda()
        img = Variable(img).cuda()

        # ===================forward=====================
        # print(f"Img gray: {img_gray.max()}")
        output = model(img_gray, img.to("cuda"))

        # loss = criterion(to_img(output), to_img(img.cuda()))
        loss = criterion(output.to("cuda"), img.to("cuda"))

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data

    # ===================log========================
    dic_losses[epoch]=total_loss

    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, total_loss))
    
        pic = to_img(output.cpu().data)
        # plt.imshow(pic[0].transpose(0,2))
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
        torch.save(model.state_dict(), f'./models/color_network_test_{epoch}.pth')

torch.save(model.state_dict(), './models/color_network_test_end.pth')

# df_losses = pd.DataFrame.from_dict(dic_losses)
# df_losses.to_csv("losses_model.csv")