__author__ = 'lstival'
# Logger
import comet_ml

import os
import torch
from torch import nn
from torchvision.utils import save_image
import kornia as K
from utils import *
# import comet_ml
from tqdm import tqdm

import read_data as ld
from architectures.ViT import *
from architectures.swin_unet import Swin_Unet
from architectures.flow import Flow_Network
from old.edge_detection import *

# ================ Importing Losses ========================
from architectures_losses.losses import ContentLoss, StyleLoss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from piq import SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from architectures_losses.vgg_loss import VGGLoss
import torchvision.models as models
import argparse as ap

def validation(dataloader_val, model, color_network, criterions, swin_model):
    total_loss = 0
    color_network.eval()
    model.eval()
    for data in dataloader_val:
        img, img_gray, img_color = create_samples(data)

        img = img.to(device)
        img_gray = img_gray.to(device)
        img_color = img_color.to(device)

        # Output of the models
        labels = color_network(img_color)
        outputs = model(img_gray, labels, swin_model)

        # Loss
        criterions[0].to(device)
        loss = criterions[0](to_img(outputs), to_img(img))
        total_loss += loss.item()

    return total_loss/len(dataloader_val)

# ================ Train Loop =====================
def train(dataroot, dataroot_val, criterions, batch_size, num_epochs, pretrained=None):
    # ================ Read Data =====================
    dataLoader = ld.ReadData()
    
    dataloader = dataLoader.create_dataLoader(dataroot, 
                        image_size, batch_size, True, constrative=False, train=True)

    # dataloader_val = dataLoader.create_dataLoader(dataroot_val,
    #                         image_size, batch_size, True, constrative=False)

    # Define the pretrained Encoder
    swin_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).to("cuda").features

    # Define the pretrained Flow Network
    flow_model = Flow_Network().to("cuda")
    flow_model.eval()
    
    # Define the model
    model = Swin_Unet(net_dimension=ch_deep, c_out=3, img_size=image_size).to(device)

    #Check pretraines
    # if pretrained is not None:
    #     resume(model, os.path.join("models", pretrained, "color_network.pth"))

    model.train()

    color_network = Vit_neck().to(device)
    color_network.eval()

    # selection of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    val_loss = 99

    # Loop over epochs
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Epochs", leave=True)
    for epoch in epoch_pbar:
        total_loss = 0
        avg_vloss_train = 0
        for idx, data in enumerate(dataloader):
            img, img_gray, img_color, next_frame = create_samples(data)

            img = img.to(device)
            img_gray = img_gray.to(device)
            img_color = img_color.to(device)
            next_frame = next_frame.to(device)

            # Output of the models
            # Color information about the image reference image 
            labels = color_network(img_color)
            # Video flow between the grayscale image and the next frame
            flow = flow_model(img_gray, next_frame)
            # Output of the model (Colorized Image)
            outputs = model(img_gray, labels, flow, swin_model)

            # Loss
            if len(criterions) == 1:
                criterions[0].to(device)
                loss = criterions[0]((outputs), (img))
                total_loss += loss.item()

            if len(criterions) == 2:
                criterions[0].to(device)
                loss = criterions[0]((outputs), (img))
                total_loss += loss.item()

                # Loss 2
                criterions[1].to(device)
                loss_2 = criterions[1](to_img(outputs), to_img(img))
                total_loss += loss_2.item()

            if len(criterions) == 3:
                criterions[0].to(device)
                loss = criterions[0]((outputs), (img))
                total_loss += loss.item()

                # Loss 2
                criterions[1].to(device)
                loss_2 = criterions[1](to_img(outputs), to_img(img))
                total_loss += loss_2.item()

                # Loss 3
                criterions[2].to(device)
                loss_3 = criterions[2](to_img(outputs), to_img(img))
                total_loss += loss_3.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

            if avg_vloss_train == 0:
                avg_vloss_train = total_loss

            avg_vloss_train = total_loss / (idx + 1)

            epoch_pbar.set_postfix(MSE=loss.item(), Val_loss = val_loss, Median=avg_vloss_train.item(),
                                    lr=optimizer.param_groups[0]['lr'])
            
            experiment.log_metrics({"loss": loss})

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

            del outputs, img, img_gray, img_color

            #Calculate validation loss
            # val_loss = validation(dataloader_val, model, color_network, criterions, swin_model)
            
            # if is_notebook:
            #     plot_images(pic[:5])
            # plt.imshow(pic[0].transpose(0,2))
            # plt.show()
            path_save_image =  f'./dc_img/{str(dt_str)}/image_{epoch}.png'
            save_image(pic, path_save_image)
            # save the image at the experiment
            # experiment.log_image( f'./dc_img/{str(dt_str)}/image_{epoch}.png', name=f"{str(dt_str)}_{epoch}")
            torch.save(model.state_dict(), f'./models/{str(dt_str)}/color_network.pth')

# Read the args from to the file
# Loss function
LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
PSNR = PeakSignalNoiseRatio()
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIMLoss(data_range=1.)
PERCEPTUAL = VGGLoss(model='vgg19')
# FID = FrechetInceptionDistance(feature=64)
VIF = VIFLoss()
VIS = VSILoss()
CONTENT = ContentLoss
STYLE = StyleLoss


# Criterions
criterions = [
[MSE],
[MAE],
[MAE , SSIM],
[MSE , SSIM],
[MAE , CONTENT],
[MSE , CONTENT],
[MAE , PSNR],
[MSE , PSNR],
[MSE , LPIPS],
[MAE , LPIPS],
[MAE , PERCEPTUAL],
[MSE , PERCEPTUAL],
[MAE , SSIM , PERCEPTUAL],
[MSE , SSIM , PERCEPTUAL],
[MSE , LPIPS , SSIM],
[MAE , LPIPS , SSIM],
[MAE , LPIPS , CONTENT],
[MSE , LPIPS , CONTENT],
[MAE , SSIM , LPIPS],
[MSE , SSIM , LPIPS],
[MAE , SSIM , STYLE],
[MSE , SSIM , STYLE]
]

parser = ap.ArgumentParser()
parser.add_argument("--criterion", type=list, default=[0])
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=101)
parser.add_argument("--model_deep", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--ch_deep", type=int, default=128)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--dataroot", type=str, default="C:/video_colorization/data/train/DAVIS")
parser.add_argument("--dataroot_val", type=str, default="C:/video_colorization/data/train/DAVIS_val")
parser.add_argument("--dt_str", type=str, default=get_model_time())

args = parser.parse_args()

# Hyper parameters
image_size = args.image_size
batch_size = args.batch_size
num_epochs = args.num_epochs
model_deep = args.model_deep
learning_rate = args.learning_rate
ch_deep = args.ch_deep
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt_str = args.dt_str
pretrained = args.pretrained
dataroot = args.dataroot
dataroot_val = args.dataroot_val
criterion_idx = int(args.criterion[0])

comet_logger = comet_ml.Experiment(
api_key="OQKd3iAY8RV0sgnY9VNde2D3G",
project_name="loss_comparison",
log_code=True)

experiment = comet_logger

comet_logger.log_parameters({
    "batch_size": batch_size,
    "img_size": image_size,
    "used_dataset": "DAVIS",
    "coment_logger": "many loss functions tested",
    # "lr": 1e-4,
    # "weight_decay": 1e-6,
    "scheduler": "CosineAnnealingLR",
    # "loss": "SSIMLoss + MSE",
    "optimizer": "AdamW",
    })

print(criterion_idx)

train(dataroot, dataroot_val, criterions[criterion_idx], batch_size, num_epochs, pretrained)

if __name__ == "__main__":
    # Loss function
    LPIPS = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
    PSNR = PeakSignalNoiseRatio()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    SSIM = SSIMLoss(data_range=1.)
    PERCEPTUAL = VGGLoss(model='vgg19')
    # FID = FrechetInceptionDistance(feature=64)
    VIF = VIFLoss()
    VIS = VSILoss()

    # Hyper parameters
    image_size = 256
    batch_size = 3
    num_epochs = 101
    model_deep = 64
    learning_rate = 2e-4
    ch_deep=128

    criterions = [MSE, LPIPS, PERCEPTUAL]

    dt_str = get_model_time()
    dt_str = "swin_unet_"+dt_str
    # comet_ml.init(project_name="swin_unet_colorization", log_code=True, log_graph=True)

    # random seed to torch 
    torch.manual_seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pretrained="swin_unet_20230621_205446"
    pretrained=None

    # Root directory for datase
    used_dataset = "DAVIS"
    val_used_dataset = "mini_DAVIS_val"

    dataroot = f"C:/video_colorization/data/train/{used_dataset}"
    dataroot_val = f"C:/video_colorization/data/train/{val_used_dataset}"

    train(dataroot, dataroot_val, criterions, batch_size, num_epochs, pretrained)
