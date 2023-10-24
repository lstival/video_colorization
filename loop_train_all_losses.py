import os
from tqdm import tqdm
from utils import *

# Losses
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from piq import SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from architectures_losses.vgg_loss import VGGLoss
from architectures_losses.losses import ContentLoss, StyleLoss

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

criterions = [
# [MSE],
# [MAE],
[MAE , SSIM],
[MSE , SSIM],
# [MAE , CONTENT],
# [MSE , CONTENT],
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
# [MAE , LPIPS , CONTENT],
# [MSE , LPIPS , CONTENT],
[MAE , SSIM , LPIPS],
[MSE , SSIM , LPIPS],
# [MAE , SSIM , STYLE],
# [MSE , SSIM , STYLE]
]

# Parameters
image_size = 256
batch_size = 6
num_epochs = 51
model_deep = 32
learning_rate = 2e-4
ch_deep=128
pretrained = None
dataroot = "C:/video_colorization/data/train/DAVIS"
dataroot_val = "C:/video_colorization/data/train/DAVIS_val"

# Loop over criterions
pbar = tqdm(range(len(criterions)))
for criterion in pbar:
    dt_str = get_model_time()
    criterion_name = get_criterion_name(criterions[criterion])
    pbar.set_description(f"Criterion: {criterion_name}")
    os.system(f"python train.py --criterion {criterion} --batch_size {batch_size} --num_epochs {num_epochs} --model_deep {model_deep} --learning_rate {learning_rate} --ch_deep {ch_deep} --pretrained {pretrained} --image_size {image_size} --dataroot {dataroot} --dataroot_val {dataroot_val} --dt_str {dt_str}")
