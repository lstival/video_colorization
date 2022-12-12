# https://huggingface.co/google/vit-base-patch32-224-in21k

# from transformers import ViTFeatureExtractor, ViTModel
# from PIL import Image
# import torch

# class ViT():
#     def __init__(self) -> None:

#         self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch32-224-in21k')
#         self.model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
#         self.model.to("cuda")
        
#     def forwad(self, image):
        
#         self.inputs = self.feature_extractor(images=image, return_tensors="pt")
#         outputs = self.model(**self.inputs)
#         last_hidden_state = outputs.last_hidden_state

#         return last_hidden_state.to("cuda")

#     print(f"Vit using Cuda?: {torch.cuda.is_available()}")


# https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch
import torch
from torch import nn
from vit_pytorch import ViT
from vit_pytorch import SimpleViT
from vit_pytorch.pit import PiT
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small

# https://github.com/lucidrains/vit-pytorch

class Vit_neck(nn.Module):
    def __init__(self, batch_size, image_size, out_chanels, device="cuda"):
        super(Vit_neck, self).__init__()

        self.v = ViT(
            image_size = image_size,
            patch_size = batch_size,
            num_classes = out_chanels,
            dim = 256,
            depth = 4,
            heads = 32,
            mlp_dim = 1024,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # self.v = SimpleViT(
        #     image_size = image_size,
        #     patch_size = batch_size,
        #     num_classes = out_chanels,
        #     dim = 1024,
        #     depth = 8,
        #     heads = 16,
        #     mlp_dim = 1024
        # )

        # self.v = PiT(
        #     image_size = image_size,
        #     patch_size = batch_size,
        #     dim = 256,
        #     num_classes = out_chanels,
        #     depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )

        # self.v = ViT_small(
        #     image_size = image_size,
        #     patch_size = batch_size,
        #     num_classes = out_chanels,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )

        self.device = device

    def forward(self, x) -> torch.Tensor:
        self.v.to(self.device)
        preds = self.v(x)
        return preds


# img = torch.randn(1, 3, 256, 256)

# preds = v(img) # (1, 1000)
# print(preds.shape)
