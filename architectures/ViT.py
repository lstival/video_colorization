import torch
from torch import nn
from torchvision.models import vit_b_32

#Feature exctration from Vit Pytorch
# https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/3

class Vit_neck(nn.Module):
    def __init__(self):
        super(Vit_neck, self).__init__()

        self.v = vit_b_32(weights="ViT_B_32_Weights.IMAGENET1K_V1").to("cuda")
        feature_exctration = torch.nn.Sequential(*(list(self.v.children())[:-1]))
        self.conv = feature_exctration[0]
        self.encoder = feature_exctration[1]

    def forward(self, x) -> torch.Tensor:
        # Process image in input
        x = self.v._process_input(x)
        # Get the number of samples
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.v.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # Get the features from image 50x768
        x = self.encoder(x)
        return x[:, 1:, :]
    
if __name__ == '__main__':
    image_size = 224
    batch_size = 2

    img = torch.ones((batch_size,3,image_size,image_size)).to("cuda")
    model = Vit_neck().to("cuda")
    out = model(img)
    
    # out shape = from image (bx50x768)
    print(out.shape)