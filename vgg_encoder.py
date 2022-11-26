
from torchvision import models
from torch import nn
import torch

# https://pytorch.org/vision/master/models/generated/torchvision.models.vgg16.html

vgg = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

class VGG_encoder(nn.Module):

    def __init__(self) -> None:
        super(VGG_encoder, self).__init__()
        # self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # self.vgg_encoder = nn.Sequential(self.vgg.features)
        self.vgg_encoder = nn.Sequential(*list(vgg.features.children()))

    def forward(self, images):
        return self.vgg_encoder(images)
    
    # def get_features(name):
    #     def hook(model, input, output):
    #         features[name] = output.detach()
    #     return hook


# Get intermediate layers features (outputs)
# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html