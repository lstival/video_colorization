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
from vit_pytorch import ViT

class Vit_neck():
    def __init__(self, batch_size, image_size, out_chanels):
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

    def forward(self, x) -> torch.Tensor:
        self.v.to("cuda")
        preds = self.v(x)
        return preds


# img = torch.randn(1, 3, 256, 256)

# preds = v(img) # (1, 1000)
# print(preds.shape)