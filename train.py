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

#to create the timestamp
from datetime import datetime
dt = datetime.now()
# dt_str = datetime.timestamp(dt)

dt_str = str(dt).replace(':','.')
dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')

#  dataloader class
# import DAVIS_dataset as ld
import load_data as ld

# vision transform
from ViT import *

from edge_detection import *

# random seed to torch 
torch.manual_seed(2021)

def save_losses(dic_losses, filename="losses_network_v1"):
    os.makedirs("losses", exist_ok=True)
    fout = f"losses/{filename}_{str(dt_str)}.csv"
    fo = open(fout, "w")

    for k, v in dic_losses.items():
        fo.write(str(k) + "," +str(float(v.cpu().numpy())) + '\n')

    fo.close()

device = "cuda"

# Images of training epochs
os.makedirs(f'./dc_img/{str(dt_str)}', exist_ok=True)

# Models Save
os.makedirs(f'./models/{str(dt_str)}', exist_ok=True)

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x  

# ================ Read Data =====================
dataLoader = ld.ReadData()

# Root directory for datase

# dataroot = "C:/video_colorization/data/train_10k_animations"
# dataroot = "C:/video_colorization/data/train/COCO_val2017"
# dataroot = "C:/video_colorization/data/train/charades"
# dataroot = "C:/video_colorization/data/train/sunset"
# dataroot_val = "C:/video_colorization/data/train/sunset_val"
# dataroot = "C:/video_colorization/data/train/diverses"
dataroot = "C:/video_colorization/data/DAVIS"
dataroot_val = "C:/video_colorization/data/DAVIS_val"

# Spatial size of training images. All images will be resized to this
image_size = (256, 256)

# https://neptune.ai/blog/pytorch-loss-functions

# Batch size during training
batch_size = 32
num_epochs = 101

dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, True)
dataloader_val = dataLoader.create_dataLoader(dataroot_val, image_size, batch_size, False)

# from color_model_vgg import ColorNetwork
from color_model import ColorNetwork

## Model setings
model = ColorNetwork(in_channel=1, out_channel=128, stride=2, padding=2, img_size=image_size[0]).to(device)

# ================ Losses ========================

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import ssim, SSIMLoss
from torch.nn import KLDivLoss
from vgg_loss import VGGLoss

LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
PSNR = PeakSignalNoiseRatio()
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIMLoss(data_range=1.)

# Falta Vram
# PERCEPTUAL = VGGLoss(model='vgg19')
# FID = FrechetInceptionDistance(feature=64)

# ================ Train ======================== 

# learning rate of the netowrk
learning_rate = 1e-2

dic_losses = {}

# define the loss
criterion = MSE # Apply in the output
criterion_2 = SSIM # Apply in image from output

criterion_2.to(device)
criterion.to(device)

# selection of the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

def create_samples(data):
    img, img_color = data
    img.to(device)
    img_gray = transforms.Grayscale(num_output_channels=1)(img).to(device)

    img_gray = Variable(img_gray).to(device)
    img = Variable(img).to(device)

    return img, img_gray, img_color

best_vloss = 1000000

# loop to train
for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        """
        img: Image with RGB colors
        img_gray: Grayscale version of the img (this) variable will be used to be colorized
        img_color: the image with color that bt used as example
        """
        img, img_gray, img_color = create_samples(data)

        model.train()
        # ===================forward=====================
        output = model(img_gray, img.to(device))

        loss1 = criterion_2(to_img(output), to_img(img.to(device)))
        loss2 = criterion(output.to(device), img.to(device))

        loss = loss2 + loss1

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data

    # ===================log========================
    dic_losses[epoch]=total_loss

    if total_loss < best_vloss:
        best_vloss = total_loss
        model_path = './models/{}/color_network_{}.pth'.format(str(dt_str), epoch)
        torch.save(model.state_dict(), model_path)

    if epoch % 10 == 0:
        # ===================validation====================
        with torch.no_grad():
            model.eval()
            running_vloss = 0.0
            for i, vdata in enumerate(dataloader_val):
                img, img_gray, img_color = create_samples(vdata)
                voutputs = model(img_gray, img.to(device))
                vloss1 = criterion_2(to_img(voutputs), to_img(img))
                vloss2 = criterion(voutputs.to(device), img.to(device))
                vloss = vloss1+vloss2
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('Epoch [{}/{}] LOSS train {:.4f} valid {:.4f}'.format(epoch+1, num_epochs, total_loss, avg_vloss))

            pic = to_img(output.cpu().data)
            # plt.imshow(pic[0].transpose(0,2))
            save_image(pic, f'./dc_img/{str(dt_str)}/image_{epoch}.png')
            # torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network_{epoch}.pth')


torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network.pth')

#Save the losses of the model
save_losses(dic_losses, f"losses_network_{dt_str}")

print("Training Finish")