from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torch.nn as nn
import torch

class Flow_Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Load the flow preprocessing
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()

        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        self.model = self.model.eval()

    def forward(self, x, x_next):
        # Preprocess the images
        x, x_next = self.transforms(x, x_next)
        # Get the flow
        flow = self.model(x, x_next)

        return flow[-1]
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from torchvision import transforms

    # Load the images
    img_batch = Image.open(r"C:\video_colorization\data\train\DAVIS\bus\00062.jpg")
    img0_batch = Image.open(r"C:\video_colorization\data\train\DAVIS\bus\00063.jpg")

    img_batch = transforms.ToTensor()(img_batch)
    img0_batch = transforms.ToTensor()(img0_batch)

    img1_batch = torch.stack([img_batch, img0_batch], dim=0)
    img2_batch = torch.stack([img0_batch, img_batch], dim=0)

    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[128, 128], antialias=False)
        img2_batch = F.resize(img2_batch, size=[128, 128], antialias=False)
        return img1_batch, img2_batch

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    # Calculate de flow network

    # If you can, run this example on a GPU, it will be a lot faster.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"

    model = Flow_Network().to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

    # Plot the flow
    plt.rcParams["savefig.bbox"] = "tight"

    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

    from torchvision.utils import flow_to_image

    predicted_flows = list_of_flows

    flow_imgs = flow_to_image(predicted_flows)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    plot(grid)