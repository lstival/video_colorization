__author__ = 'lstival'

# https://github.com/L1aoXingyu
# https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
# https://huggingface.co/models?sort=downloads


# https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import kornia as K
from utils import *

#to create the timestamp
from datetime import datetime
dt = datetime.now()
# dt_str = datetime.timestamp(dt)

dt_str = str(dt).replace(':','.')
dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# Import comet to log the model training
import comet_ml
comet_ml.init(project_name="natural_video_colorization", log_code=True, log_graph=True)

#  dataloader class
import DAVIS_dataset as ld
# import load_data as ld

# vision transform
from architectures.ViT import *

from edge_detection import *

# random seed to torch 
torch.manual_seed(2021)

def save_losses(dic_losses, filename="losses_network_v1"):
    os.makedirs("losses", exist_ok=True)
    fout = f"{filename}.csv"
    fo = open(fout, "w")

    for k, v in dic_losses.items():
        fo.write(str(k) + "," +str(float(v.cpu().numpy())) + '\n')

    fo.close()

device = "cuda"

# Images of training epochs
os.makedirs(f'./dc_img/{str(dt_str)}', exist_ok=True)

# Models Save
os.makedirs(f'./models/{str(dt_str)}', exist_ok=True)

#Path to loss
path_loss = f"losses/{dt_str}"
os.makedirs(path_loss, exist_ok=True)

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])

    return x  

# ================ Read Data =====================
dataLoader = ld.ReadData()

used_dataset = "DAVIS"
# Root directory for datase

# dataroot = "C:/video_colorization/data/train_10k_animations"
# dataroot = "C:/video_colorization/data/train/COCO_val2017"
# dataroot = "C:/video_colorization/data/train/charades"
# dataroot = "C:/video_colorization/data/train/sunset"
# dataroot_val = "C:/video_colorization/data/train/sunset_val"
# dataroot = "C:/video_colorization/data/train/diverses"

# dataroot = "C:/video_colorization/data/DAVIS"
# dataroot_val = "C:/video_colorization/data/DAVIS_val"

dataroot = f"C:/video_colorization/data/train/{used_dataset}"
dataroot_val = f"C:/video_colorization/data/train/{used_dataset}_val"

# https://neptune.ai/blog/pytorch-loss-functions


# ================ Train Parans ========================
# Spatial size of training images. All images will be resized to this
image_size = (128, 128)

# Batch size during training
batch_size = 45
num_epochs = 101
model_deep = 128

learning_rate = 1e-2

# Number min number of channels in the model
ch_deep=40
# hyper_params = {"batch_size": batch_size, 
#         "num_epochs": num_epochs, 
#         "learning_rate": learning_rate}

experiment = comet_ml.Experiment(
    api_key="SNv4ks5JjUxZ1X0FhARDGt4SY",
    project_name="natural_video_colorization",
    workspace="lstival")

dataloader = dataLoader.create_dataLoader(dataroot, 
                        image_size, batch_size, True)

dataloader_val = dataLoader.create_dataLoader(dataroot_val,
                        image_size, batch_size, False)

# from color_model_vgg import ColorNetwork
# from color_model import ColorNetwork
from architectures.color_model_simple import ColorNetwork

## Model setings
model = ColorNetwork(1,3,image_size[0],ch_deep=ch_deep).to(device)
# model = ColorNetwork(in_channel=1, out_channel=model_deep, 
#                     stride=2, padding=2, img_size=image_size[0]).to(device)

# selection of the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
            lr=learning_rate, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# ================ Losses ========================

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from torch.nn import KLDivLoss
from architectures_losses.vgg_loss import VGGLoss
from architectures_losses.smooth_loss import Loss
from pytorch_tools import losses

LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
PSNR = PeakSignalNoiseRatio()
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIMLoss(data_range=1.)
PERCEPTUAL = VGGLoss(model='vgg19')
FID = FrechetInceptionDistance(feature=64)
VIF = VIFLoss()
VIF = VSILoss()
STYLE = losses.StyleLoss(model="vgg19_bn", layers=["21"])
CONTENT = losses.ContentLoss(model="vgg19_bn", layers=["21"]) #https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/losses/vgg_loss.py

# Falta Vram


# learning rate of the netowrk
dic_losses = {}

# define the loss
criterion = MSE # Apply in the output
criterion_2 = LPIPS # Apply in image from output
criterion_3 = CONTENT # Apply in the output (original image and first frame of scene)

criterion.to(device)
criterion_2.to(device)
criterion_3.to(device)

# ================ Logs ========================
params = {
    "batch_size": batch_size,
    "epochs": num_epochs,
    "img_size": image_size,
    "output_layers": model_deep,
    "layer1_activation": 'relu',
    "learning_rate": learning_rate,
    "weight_decay": 1e-5,
    "dataset": used_dataset,
    "optimizer": str(type(optimizer)).split('.')[-1].split("'")[0],
    # "criterion": str(type(criterion)).split('.')[-1].split("'")[0],
    # "criterion_2": str(type(criterion_2)).split('.')[-1].split("'")[0],
    # "criterion_3": str(type(criterion_3)).split('.')[-1].split("'")[0],
    "vit": "ViT",
    "network": "simple_color_model",
    "ch_deep": ch_deep,
    "comment": "",
}

experiment.log_parameters(params)

# ================ Train ======================== 

def create_samples(data):
    """
    img: Image with RGB colors
    img_gray: Grayscale version of the img (this) variable will be used to be colorized
    img_color: the image with color that bt used as example (first at the scene)
    """
    img, img_color, next_frame = data
    # img.to(device)
    # img_color.to(device)

    img_gray = transforms.Grayscale(num_output_channels=1)(img)
    # img_gray = img[:,:1,:,:]

    img = img.to(device)
    img_gray = img_gray.to(device)
    img_color = img_color.to(device)
    next_frame = next_frame.to(device)

    return img, img_gray, img_color, next_frame

best_vloss = 1000000

# loop to train
for epoch in range(num_epochs):
    total_loss = 0
    for idx, data in enumerate(dataloader):

        img, img_gray, img_color, next_frame = create_samples(data)

        model.train()
        # ================== Forward ====================
        output = model(img_gray, img_color)

        # criterios = [criterion, criterion_2, criterion_3]
        criterios = [criterion, criterion_2, criterion_3]

        loss, dict_losses = model_losses(criterios, [output, to_img(img), to_img(next_frame), to_img(output)])

        for key, value in dict_losses.items():
            commet_log_metric(experiment, key, value, epoch)

        # ================== Backward ===================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data

        avg_vloss_train = total_loss / (idx + 1)

        experiment.log_metric("avg_loss_train", avg_vloss_train, step=epoch)
        experiment.log_metric("total_loss_train", total_loss, step=epoch)

    # ================== Log =======================
    dic_losses[epoch]=avg_vloss_train

    if avg_vloss_train < best_vloss:
        best_vloss = avg_vloss_train
        model_path = './models/{}/color_network_{}.pth'.format(str(dt_str), epoch)
        torch.save(model.state_dict(), model_path)
        #Save the losses of the model
        save_losses(dic_losses, f"{path_loss}/losses_network_epoch_{epoch}")


    with torch.no_grad():
        model.eval()
        running_vloss = 0.0
        for i, vdata in enumerate(dataloader_val):
            img, img_gray, img_color, next_frame = create_samples(vdata)
            voutputs = model(img_gray, img_color)

            vloss, dict_losses_val  = model_losses(criterios, [voutputs, to_img(img), to_img(next_frame), to_img(voutputs)])

            for key, value in dict_losses_val.items():
                commet_log_metric(experiment, key, value, epoch, me_type="val")

            # vloss = vloss1+vloss2+vloss3
            running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            experiment.log_metric("avg_loss_val", avg_vloss, step=epoch)

    if epoch % 10 == 0:
        # ================== Validation ===================
            print('Epoch [{}/{}] AVG LOSS train {:.4f} || AVG valid {:.4f}'.format(epoch+1, num_epochs, avg_vloss_train, avg_vloss))

            pic = to_img(output.cpu().data)
            # plt.imshow(pic[0].transpose(0,2))
            path_save_image =  f'./dc_img/{str(dt_str)}/image_{epoch}.png'
            save_image(pic, path_save_image)
            # save the image at the experiment
            experiment.log_image( f'./dc_img/{str(dt_str)}/image_{epoch}.png', name=f"{str(dt_str)}_{epoch}")
            # torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network_{epoch}.pth')


torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network.pth')

#Save the losses of the model
save_losses(dic_losses, f"{path_loss}/losses_network_end")

print("Training Finish")
experiment.end()

# https://github.com/qubvel/segmentation_models.pytorch#models