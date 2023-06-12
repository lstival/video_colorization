"""
Evaluation of the model, and save the video colorized
from the example image passed
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image, ImageOps
import cv2
from torch.autograd import Variable

from torchvision import io
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import DAVIS_dataset as ld

from utils import *

import shutil
from architectures.color_model_simple import ColorNetwork

dataLoader = ld.ReadData()

image_size = (128, 128)
device = "cuda"
# video_class = "parkour"
str_dt = "20221223_160049"
# dataset = "DAVIS_val"
dataset = "videvo"
batch_size = 1

# List all classes to be evaluated
images_paths = f"C:/video_colorization/data/train/{dataset}"
img_classes = os.listdir(images_paths)

# def create_samples(data):
#     """
#     img: Image with RGB colors
#     img_gray: Grayscale version of the img (this) variable will be used to be colorized
#     img_color: the image with color that bt used as example (first at the scene)
#     """
#     img, img_color, _ = data
#     img_gray = transforms.Grayscale(num_output_channels=1)(img)
#     imgs_2.append(img)
#     imgs_2_gray.append(img_gray)

#     img.to(device)
#     img_gray.to(device)
#     img_color.to(device)

#     return img, img_gray, img_color


# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

# ================ Read Data ===================
#recives a video and a example color image
# path = "C:/video_colorization/data/videos/woman_gray.mp4"

# frames, _, _ = io.read_video(str(path), output_format="TCHW")

root_model_path = "E:/models"
avaliable_models = os.listdir(root_model_path)

pbar = tqdm(avaliable_models)
# ================ Loop all videos inside gray folder =====================
for str_dt in pbar:
    pbar.set_description(f"Processing: {str_dt}")

    temp_path = f"temp/{dataset}/"

    # if os.path.exists(temp_path):
    #     shutil.rmtree(temp_path)

    os.makedirs(temp_path, exist_ok=True)

    # ================ Read Video =====================
    path_gray_video = f"C:/video_colorization/data/videos/{dataset}_gray/"
    try:
        list_gray_videos = os.listdir(path_gray_video)
    except FileNotFoundError:
        # Create the gray version of the videos
        create_gray_videos(dataset, path_gray_video)
        list_gray_videos = os.listdir(path_gray_video)

    # ================ Read Model =====================
    # predict the color for each frame in the video
    # model_path = f'./models/{str_dt}/color_network.pth'
    model_path = f'{root_model_path}/{str_dt}/color_network.pth'

    # try:
    model = load_trained_model(model_path, image_size, device)
    # except:
    #     model = load_trained_model(model_path, image_size, device, ch_deep=40)
    # except:
    #     model = load_trained_model(model_path, image_size, device, ch_deep=128)
    

    pbar = tqdm(list_gray_videos)
    # ================ Loop all videos inside gray folder =====================
    for video_name in pbar:
        pbar.set_description(f"Processing: {video_name}")

        vidcap = cv2.VideoCapture(f"{path_gray_video}{video_name}")
        success,image = vidcap.read()
        count = 0
        list_frames = []

        path_temp_gray_frames = f"{temp_path}{video_name.split('.')[0]}"
        if not os.path.exists(path_temp_gray_frames):

            os.makedirs(f"{path_temp_gray_frames}/images/", exist_ok=True)

            while success:
                cv2.imwrite(f"{path_temp_gray_frames}/images/{str(count).zfill(5)}.jpg", image)     # save frame as JPEG file      
                success,image = vidcap.read()
                list_frames.append(image)
                count += 1
        # print("Finsh")


    # ================ Read images to make the video =====================
        dataloader = dataLoader.create_dataLoader(path_temp_gray_frames, image_size, batch_size)

        example_path = f"C:/video_colorization/data/train/{dataset}/{video_name}/"

        # example_img = Image.open(f"{example_path}{str(count-1).zfill(5)}.jpg", )
        example_img = Image.open(f"C:/video_colorization/data/train/{dataset}/{video_name.split('.')[0]}/00010.jpg")
        example_img = transforms.functional.pil_to_tensor(example_img.resize(image_size)).to(device)
        example_img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(example_img.type(torch.float32))


    # ============== Frame Production ===================
        imgs_2 = []
        imgs_2_gray = []

        outs = []
        # path to save colored frames
        colored_frames_save = f"temp_result/{dataset}/{str_dt}/{video_name}/"

        if os.path.exists(colored_frames_save):
            shutil.rmtree(colored_frames_save)

        os.makedirs(colored_frames_save, exist_ok=True)

        # path so save videos
        colored_video_path = f"videos_output/{str_dt}/{video_name}/"
        os.makedirs(colored_video_path, exist_ok=True)

        with torch.no_grad():
            model.eval()
            imgs_data = []

            for idx, data in enumerate(dataloader):

                img, img_gray, img_color = create_samples(data)
                imgs_data.append(img_gray)

                # img_frame = transforms.Grayscale(num_output_channels=1)(img_frame)
                out = (model(img_gray.to(device), example_img.unsqueeze(0))).cpu()
                outs.append(out)

            for idx, frame in enumerate(outs):
                save_image(to_img(frame), f"{colored_frames_save}{str(idx).zfill(5)}.jpg")

        frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4")

    # aa = model(gray_cow.unsqueeze(0), example_img.unsqueeze(0)).cpu()
    # save_image(to_img(aa), f"{str(999).zfill(5)}.jpg")

print("Evaluation Finish")

# plt.imshow(to_img(outs[0])[0].transpose(0,2).cpu().detach().numpy())
# ============== Reconscruct ===================
# reconscruct the video from frames colorizeds
# frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4")
