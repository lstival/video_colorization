import numpy as np
import cv2 as cv
import os
from PIL import Image
import glob 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

# list of images
imgs = []

# ============= Read images ===============

path = "C:/video_colorization/data/japan"

dataset = dset.ImageFolder(root=path,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

# #read all imagens and put in the list
# for img in glob.glob(f"{path}/**/*.jpg"):
    # imgs.append(Image.open(img))


# for file in os.listdir(path):
#     for img in os.listdir(f"{path}/{file}"):
#         imgs.append(Image.open(f"{path}/{file}/{img}"))

# ============= Edge Dectetion ============

# list of edges    
edges = []

bound_up = 80
bound_dw = 50
aperture_size = 5
    
# find all edges in the images
for img in tqdm.tqdm(dataset):
    temp_img = np.uint8(img[0]*255)
    temp_img = np.moveaxis(temp_img, 0, 2)
    edges.append(cv.Canny(temp_img, bound_dw, bound_up))

# ============= Save Images ===============

path_2_save = "C:/video_colorization/data/train/train_edges/images"

os.makedirs(path_2_save, exist_ok=True)

for count, img in tqdm.tqdm(enumerate(edges)):
    PIL_image = Image.fromarray(img)
    PIL_image.save(f"{path_2_save}/{count}.jpg")