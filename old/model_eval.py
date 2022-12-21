import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from color_model import *

# ================ Read Model =====================

path = f'./models/color_netowrk_end60.pth'

## Model setings
model = ColorNetwork(in_channel=1, out_channel=32, stride=2, padding=2).cuda()

# learning rate of the netowrk
learning_rate = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                            weight_decay=1e-6)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint)

model.eval()

# ================ Read Data =====================
#  dataloader class
import load_data as ld

dataLoader = ld.ReadData()

# Root directory for dataset
dataroot = "C:/video_colorization/data/train/woman"

# Spatial size of training images. All images will be resized to this
image_size = (256, 256)

# Batch size during training
batch_size = 64

dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size)

# ================ Evaluation =====================
# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

img = next(iter(dataloader))
img = img[0]
img_gray = transforms.Grayscale(num_output_channels=1)(img)

out = model(img_gray.cuda(), img.cuda())

out2 = np.uint8(out[0].cpu().detach().numpy()*255)
out2 = np.moveaxis(out2, 0, 2)

plt.imshow(out2)