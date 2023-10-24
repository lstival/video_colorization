import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vit_b_32, vgg19_bn
from torchvision.models import VGG19_BN_Weights

#Feature exctration from Vit Pytorch
# https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/3

class Vit_neck(nn.Module):
    def __init__(self):
        super(Vit_neck, self).__init__()

        # self.v = vit_b_32(weights="ViT_B_32_Weights.IMAGENET1K_V1").to("cuda")
        feature_exctration = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).to("cuda")
        self.v = feature_exctration.features

    def forward(self, x) -> torch.Tensor:
        #Reshape the image to 256x256
        x = nn.functional.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)

        # Process image in input
        self.v.eval()
        x = self.v(x)
        return x
    
if __name__ == '__main__':
    image_size = 512
    batch_size = 2

    img = torch.ones((batch_size,3,image_size,image_size)).to("cuda")
    model = Vit_neck().to("cuda")
    # model = vgg19_bn(weights="VGG19_BN_Weights.IMAGENET1K_V1").to("cuda")
    # out = torch.nn.Sequential(*(list(model.children())[:-1]))(img)
    out = model(img)

    # out shape = from image (bx50x768)
    print(out.shape)