import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

path = "C:/video_colorization/data/train/DAVIS"
path_gray = "C:/video_colorization/Vit-autoencoder/temp"

class DAVISDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size):
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size

        self.scenes = os.listdir(path)
        self.color_examples, self.samples = self.__getscenes__(self.path)

    def __getscenes__(self, path):
        """
        Return two list: One with the samples the other
        with the first frame in the folder
        """
        color_examples = []
        samples = []

        for scene in self.scenes:
            scene_path = f"{path}/{scene}/"
            for scene_frame in range(len(os.listdir(scene_path))-1):
                color_examples.append(self.__transform__(Image.open(f"{scene_path}/{str(0).zfill(5)}.jpg"))) #Img 0
                samples.append(self.__transform__(Image.open(f"{scene_path}/{str(scene_frame+1).zfill(5)}.jpg"))) # Other Imgs


        # color_examples = samples.copy()

        # assert torch.eq(samples[0], samples[10]), "Samples must be differents"
        assert len(samples) == len(color_examples)

        return samples, color_examples

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """
        transform=transforms.Compose([
                                    transforms.Resize(self.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
        x_transformed = transform(x)
        return x_transformed
        
    def __len__(self):
        """
        Return hou much samples as in the dataset.
        """
        return len(self.color_examples)
        
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        return self.color_examples[index], self.samples[index]


# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, shuffle=False):
        self.datas = DAVISDataset(dataroot, image_size)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle)

        assert (next(iter(self.dataloader))[0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader

# Loop for each scene to get the 

# cls = ReadData()
# temp_path = "temp/images"
# dataloader = cls.create_dataLoader(temp_path.split('/')[0], (128, 128), 1)

# datas = DAVISDataset(path_gray, (256, 256))

# dataloader = torch.utils.data.DataLoader(datas, batch_size=16,shuffle=False)
# print(next(iter(dataloader))[0].shape)
# print(next(iter(dataloader))[1].shape)