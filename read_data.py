import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import kornia as K
from utils import *
import random
import numpy as np

class ColorizationDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size, constrative=False, train=None):
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size
        self.constrative = constrative
        self.train = train

        self.scenes = os.listdir(path)

        self.dataset = torchvision.datasets.ImageFolder(self.path, self.__transform__)

    def __colorization_transform__(self, x):

        if self.train:
            colorization_transform=transforms.Compose([
                    torchvision.transforms.Resize(280),  # args.image_size + 1/4 *args.image_size
                    torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                    # torchvision.transforms.RandomRotation((0, 365)),
                    # torchvision.transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.image_size,self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            colorization_transform=transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        
        return colorization_transform(x)

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """

        x_transformed = self.__colorization_transform__(x)
        return x_transformed
        
    def __len__(self):
        """
        Return hou much samples as in the dataset.
        """
        return len(self.dataset)

    
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        # Get the next indices
        keyframe_index = min(index - 1, len(self.dataset))
        next_index = min(index + 1, len(self.dataset) - 1)
        
        if self.constrative:
            # random_idx = random.randint(1, 10)
            # random_idx = min(index + 1, len(self.dataset) - 1)
            # return self.dataset[index], self.dataset[keyframe_index], self.dataset[random_idx], self.dataset[next_index]
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index]
        else:
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index]
        

# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, shuffle=True, pin_memory=True, constrative=False , train=None):

        self.datas = ColorizationDataset(dataroot, image_size, constrative=constrative, train=train)

        # self.datas = DAVISDataset(dataroot, image_size, rgb=rgb, pos_path=pos_path, constrative=constrative)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        # assert (next(iter(self.dataloader))[0][0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader


if __name__ == '__main__':

    from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
    from sklearn.model_selection import KFold

    print("main")

    batch_size = 5
    used_dataset = "mini_DAVIS"

    dataroot = f"C:/video_colorization/data/train/{used_dataset}/"
    # valid_dataroot = f"C:/video_colorization/diffusion/data/latens/mini_DAVIS_val/"

    dataLoader = ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, 224, batch_size, shuffle=True, constrative=False, train=False)

    import matplotlib.pyplot as plt

    data = next(iter(dataloader))

    # img, img_gray, img_color, next_frame, pos_frame = create_samples(data, pos_imgs=True)
    img, img_gray, img_color, next_frame = create_samples(data)

    # Plt the ground truth img
    b = to_img(img)
    plot_images(b)

    # Plt gray img
    b_gray = to_img(img_gray)
    plot_images(b_gray)

    # Plot color img (example)
    b_color = to_img(img_color)
    plot_images(b_color)

    # Plot color img (example)
    c_color = to_img(next_frame)
    plot_images(c_color)