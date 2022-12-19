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

import DAVIS_dataset as ld

from utils import *

import shutil

image_size = (128, 128)

device = "cuda"

video_class = "parkour"

str_dt = "20221216_130444"

# ================ Read Data ===================
#recives a video and a example color image
# path = "C:/video_colorization/data/videos/woman_gray.mp4"

# frames, _, _ = io.read_video(str(path), output_format="TCHW")

temp_path = "temp/images"

# video_name = "C:/video_colorization/data/videos/gray/sunset_gray.mp4"
video_name = f"C:/video_colorization/data/videos/gray/video_{video_class}.mp4"

# ================ Read Video =====================
if os.path.exists(temp_path):
    shutil.rmtree(temp_path)

os.makedirs(temp_path)

vidcap = cv2.VideoCapture(f"{video_name}")
success,image = vidcap.read()
count = 0
list_frames = []


while success:
    cv2.imwrite(f"{temp_path}/{str(count).zfill(5)}.jpg", image)     # save frame as JPEG file      
    success,image = vidcap.read()
    list_frames.append(image)
    count += 1
print("Finsh")

batch_size = 1
dataLoader = ld.ReadData()

dataloader = dataLoader.create_dataLoader(temp_path.split('/')[0], image_size, batch_size)

example_path = f"C:/video_colorization/data/train/DAVIS_val/{video_class}/"

# example_img = Image.open(f"{example_path}{str(count-1).zfill(5)}.jpg", )
example_img = Image.open("C:/video_colorization/data/train/DAVIS_val/parkour/00010.jpg")
example_img = transforms.functional.pil_to_tensor(example_img.resize(image_size)).to(device)
example_img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(example_img.type(torch.float32))

# ================ Read Model =====================
# predict the color for each frame in the video

from color_model_simple import *

path = f'./models/{str_dt}/color_network.pth'

#old
# model = ColorNetwork(in_channel=1, out_channel=128, stride=2, padding=2,img_size=image_size[0]).to(device)
model = ColorNetwork(1, 3, image_size[0], ch_deep=64).to(device)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
model.to(device)

# ============== Frame Production ===================
imgs_2 = []
imgs_2_gray = []

def create_samples(data):
    """
    img: Image with RGB colors
    img_gray: Grayscale version of the img (this) variable will be used to be colorized
    img_color: the image with color that bt used as example (first at the scene)
    """
    img, img_color, _ = data
    img_gray = transforms.Grayscale(num_output_channels=1)(img)
    imgs_2.append(img)
    imgs_2_gray.append(img_gray)

    img.to(device)
    img_gray.to(device)
    img_color.to(device)

    return img, img_gray, img_color


# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

outs = []
# path to save colored frames
colored_frames_save = f"temp_result/{str_dt}/"

if os.path.exists(colored_frames_save):
    shutil.rmtree(colored_frames_save)

os.makedirs(colored_frames_save, exist_ok=True)

# path so save videos
colored_video_path = f"videos_output/{str_dt}/"
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

    # aa = model(gray_cow.unsqueeze(0), example_img.unsqueeze(0)).cpu()
    # save_image(to_img(aa), f"{str(999).zfill(5)}.jpg")

print("Evaluation Finish")

# plt.imshow(to_img(outs[0])[0].transpose(0,2).cpu().detach().numpy())
# ============== Reconscruct ===================
# reconscruct the video from frames colorizeds
frame_2_video(colored_frames_save, f"{colored_video_path}/{video_class}_colored.mp4")
