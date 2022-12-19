import cv2
import os
import torch

def read_frames(image_folder,):
    """
    Read all frames of the image_folder, salve it
    in a folder and return.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    if images == []:
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    return images

def frame_2_gray(images) -> list:
    """
    Recive a list of images and return the same list
    but with the images in gray scale.
    """

    gray_frames = []

    for img in images:
        gray_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return gray_frames

def frame_2_video(image_folder, video_name, gray=False, frame_rate=16):
    """
    Get the path with the frames and the name that video must be
    and create and save the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    images = read_frames(image_folder)
        
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    if gray == True:
        video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width,height), 0)
    else:
        video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width,height))

    for image in images:
        temp_image = cv2.imread(os.path.join(image_folder, image), 1)
        
        if gray == True:
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        video.write(temp_image)

    video.release()
    print("Convertion Done")


 ################## Losses #####################
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import ssim, SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from torch.nn import KLDivLoss
from architectures_losses.vgg_loss import VGGLoss
from architectures_losses.smooth_loss import Loss
from pytorch_tools import losses


def valid_loss(loss) -> torch.Tensor:
    """
    Recive a loss and verify if it is a torch.Tensor,
    and return a Tensor if it is a list.
    """
    if type(loss) == list:
        loss = loss[0]

    return loss

def model_losses(losses: list, inputs: list):
    """
    losses: List of loss functions to calculate the loss
    inputs: List of data to calculate the loss.

    return a sum of all losses and List contains all losses individually
    """

    dict_losses = {}

    # train_losses = []

    total_losses = 0

    # populate values of each loss
    for idx, loss in enumerate(losses):
        #loss name
        loss_name = str(type(loss)).split(".")[-1].split("'")[0]

        if len(dict_losses) == 0:
            dict_losses[loss_name] = valid_loss(loss(inputs[0], inputs[1]))
        if len(dict_losses) < 2:
            dict_losses[loss_name] = valid_loss(loss(inputs[3], inputs[1]))
        else:
            dict_losses[loss_name] = valid_loss(loss(inputs[3], inputs[2]))
            
    assert len(dict_losses) == len(losses), "Loss and Output has to have same size"
    # # create variables for each loss
    # for key, value in dict_losses.items():
    #     exec(f"{key}={value}")
    
    # calculate total loss
    for key, value in dict_losses.items():
        
        total_losses += value

    return total_losses, dict_losses

def commet_log_metric(experiment, metric_name: str, metric, step: int, me_type="train") -> None:
    """
    Generic method to create the experiment log
    """
    experiment.log_metric(f"{metric_name}_{me_type}", metric, step=step)


# Gerenate grayscale videos of all videos in DAVIS_val

# for class_video in os.listdir("C:/video_colorization/data/train/DAVIS_val"):
#     image_folder = f"C:/video_colorization/data/train/DAVIS_val/{class_video}"
#     path_video_save = "C:/video_colorization/data/videos/gray/"
#     video_name = f'{path_video_save}video_{class_video}.mp4'

#     frame_2_video(image_folder, video_name, True)


# image_folder = "C:/video_colorization/Vit-autoencoder/temp_result/20221207_111837"

# class_video = "breakdance"

# image_folder = f"C:/video_colorization/data/train/DAVIS_val/{class_video}"
# path_video_save = "C:/video_colorization/data/videos/gray/"
# video_name = f'{path_video_save}video_{class_video}.mp4'

# frame_2_video(image_folder, video_name, True)
