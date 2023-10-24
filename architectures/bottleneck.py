import torch
from torch import nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vgg16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_ft = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        out = self.model_ft(x)
        return out

v = Vgg16()
sample = torch.rand((2,3,256,256))

print(v(sample).shape)
