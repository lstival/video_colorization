import torch
from torch import nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vgg16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_ft = models.vgg16(pretrained=True)

    def forward(self, x):
        out = self.model_ft(x)
        return out

v = Vgg16()
sample = torch.rand((2,3,256,256))

print(v(sample).shape)
