__author__ = 'lstival'

import os
import torch
from torch import nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import kornia as K
from utils import *
# import comet_ml
from tqdm import tqdm

import read_data as ld
from architectures.ViT import *
from architectures.swin_unet import Swin_Unet
from edge_detection import *

# ================ Importing Losses ========================

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from torch.nn import KLDivLoss
from architectures_losses.vgg_loss import VGGLoss
from architectures_losses.smooth_loss import Loss

# ================ Train Loop =====================
def train(dataroot, dataroot_val, criterions, batch_size, num_epochs):
    # ================ Read Data =====================
    dataLoader = ld.ReadData()
    
    dataloader = dataLoader.create_dataLoader(dataroot, 
                        image_size, batch_size, True, constrative=False, train=True)

    # dataloader_val = dataLoader.create_dataLoader(dataroot_val,
    #                         image_size, batch_size, True, constrative=False)
    
    model = Swin_Unet(net_dimension=ch_deep, c_out=3, img_size=image_size).to(device)
    model.train()

    color_network = Vit_neck().to(device)

    # selection of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    # Loop over epochs
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        total_loss = 0
        for idx, data in enumerate(dataloader):
            img, img_gray, img_color = create_samples(data)

            img = img.to(device)
            img_gray = img_gray.to(device)
            img_color = img_color.to(device)

            # Output of the models
            labels = color_network(img_color)
            outputs = model(img_gray, labels)

            # Loss
            criterions[0].to(device)
            loss = criterions[0]((outputs), (img))
            total_loss += loss.item()

            # Loss 2
            criterions[1].to(device)
            loss_2 = criterions[1](to_img(outputs), to_img(img))
            total_loss += loss_2.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            avg_vloss_train = total_loss / (idx + 1)

            epoch_pbar.set_postfix(MSE=loss.item(), Median=avg_vloss_train.item(),
                                    lr=optimizer.param_groups[0]['lr'])
            
        scheduler.step()

            # experiment.log_metric("avg_loss_train", avg_vloss_train, step=epoch)
            # experiment.log_metric("total_loss_train", total_loss, step=epoch)

        if epoch % 10 == 0:
            # Images of training epochs
            os.makedirs(f'./dc_img/{str(dt_str)}', exist_ok=True)

            # Models Save
            os.makedirs(f'./models/{str(dt_str)}', exist_ok=True)

            #Path to loss
            path_loss = f"losses/{dt_str}"
            os.makedirs(path_loss, exist_ok=True)

            pic = to_img(outputs[:5].cpu().data)
            
            # if is_notebook:
            plt.imshow(pic[0].transpose(0,2))
            path_save_image =  f'./dc_img/{str(dt_str)}/image_{epoch}.png'
            save_image(pic, path_save_image)
            # save the image at the experiment
            # experiment.log_image( f'./dc_img/{str(dt_str)}/image_{epoch}.png', name=f"{str(dt_str)}_{epoch}")
            torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network.pth')

if __name__ == "__main__":

    # Hyper parameters
    image_size = 224
    batch_size = 9
    num_epochs = 201
    model_deep = 128
    learning_rate = 2e-4
    ch_deep=128

    # Loss function
    LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    PSNR = PeakSignalNoiseRatio()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    SSIM = SSIMLoss(data_range=1.)
    PERCEPTUAL = VGGLoss(model='vgg19')
    FID = FrechetInceptionDistance(feature=64)
    VIF = VIFLoss()
    VIS = VSILoss()

    criterions = [MAE, SSIM]

    # ================ Logs ========================
    # params = {
    #     "batch_size": batch_size,
    #     "epochs": num_epochs,
    #     "img_size": image_size,
    #     "output_layers": model_deep,
    #     "layer1_activation": 'relu',
    #     "learning_rate": learning_rate,
    #     "criterion": str(type(criterions[0])).split('.')[-1].split("'")[0],
    #     "criterion_2": str(type(criterions[1])).split('.')[-1].split("'")[0],
    #     # "criterion_3": str(type(criterions[2])).split('.')[-1].split("'")[0],
    #     "ch_deep": ch_deep,
    #     "comment": "",
    # }

    # experiment = comet_ml.Experiment(
    # api_key="SNv4ks5JjUxZ1X0FhARDGt4SY",
    # project_name="swint_unet_colorization",
    # workspace="lstival")
    
    # experiment.log_parameters(params)

    dt_str = get_model_time()
    dt_str = "swin_unet_"+dt_str
    # comet_ml.init(project_name="swin_unet_colorization", log_code=True, log_graph=True)

    # random seed to torch 
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Root directory for datase
    used_dataset = "DAVIS"
    val_used_dataset = "mini_DAVIS_val"

    dataroot = f"C:/video_colorization/data/train/{used_dataset}"
    dataroot_val = f"C:/video_colorization/data/train/{val_used_dataset}"

    train(dataroot, dataroot_val, criterions, batch_size, num_epochs)
    
