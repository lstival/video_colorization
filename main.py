import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
import cv2

from torchvision import io
from torchvision import transforms
from torchvision.utils import save_image

import load_data as ld

image_size = (128, 128)

device = "cuda"

str_dt = "20221205_202003"

# ================ Read Data ===================
#recives a video and a example color image
# path = "C:/video_colorization/data/videos/woman_gray.mp4"

# frames, _, _ = io.read_video(str(path), output_format="TCHW")

temp_path = "temp/images"
video_name = "C:/video_colorization/data/videos/gray/sunset_gray.mp4"

if not os.path.exists(temp_path):
    os.makedirs(temp_path)

vidcap = cv2.VideoCapture(f"{video_name}")
success,image = vidcap.read()
count = 0
list_frames = []
while success:
    cv2.imwrite(f"{temp_path}/frame{count}.jpg", image)     # save frame as JPEG file      
    success,image = vidcap.read()
    list_frames.append(image)
    count += 1
print("Finsh")

batch_size = 1
dataLoader = ld.ReadData()
dataloader = dataLoader.create_dataLoader(temp_path.split('/')[0], image_size, batch_size)

# img1_batch = torch.stack([frames[100].transpose(0,2), 
#                             frames[150].transpose(0,2)])

# plt.imshow(frames[150].transpose(0,2))
# example_img = Image.open("C:/video_colorization/data/train/woman/images/frame0.jpg")
# example_img = transforms.functional.pil_to_tensor(example_img.resize(image_size))

example_path = "C:/video_colorization/data/train/sunset/"
example_data = dataLoader.create_dataLoader(example_path, image_size, batch_size, False)
example_img = next(iter(example_data))
# print(f"example_img type: {example_img}")


# ================ Read Model =====================
# predict the color for each frame in the video

from color_model import *

path = f'./models/{str_dt}/color_network.pth'

model = ColorNetwork(in_channel=1, out_channel=128, stride=2, padding=2,img_size=image_size[0]).to(device)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
model.to(device)

# ============== Frame Production ===================
# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

outs = []
colored_frames_save = f"temp_result/{str_dt}/"
os.makedirs(colored_frames_save, exist_ok=True)

with torch.no_grad():
    model.eval()

    for img_frame, _ in dataloader:
        img_frame = transforms.Grayscale(num_output_channels=1)(img_frame)
        out = (model(img_frame.to(device), example_img[0].to(device))).cpu()
        outs.append(out)

    for idx, frame in enumerate(outs):
        save_image(to_img(frame), f"{colored_frames_save}{idx}.png")

print("Evaluation Finish")
        
# plt.imshow(to_img(outs[0])[0].transpose(0,2).cpu().detach().numpy())
# ============== Reconscruct ===================
# reconscruct the video from frames colorizeds