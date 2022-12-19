"""
Use the images genereated by the trained model and
compare with the grownd truth.

The results are from SSIM and PSNR values.
"""

from torchvision import transforms
to_tensor = transforms.PILToTensor()

from piq import ssim, psnr

import DAVIS_dataset as ld
dataLoader = ld.ReadData()
from PIL import Image

from utils import *

image_size = (128, 128)
device = "cuda"
video_class = "parkour"
str_dt = "20221216_130444"

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x


# Read frames colored by the model
colored_video_path = f"temp_result/{str_dt}/"
list_frame = read_frames(colored_video_path)

# Read the grownd truth images
original_video_path = f"C:/video_colorization/data/train/DAVIS_val/{video_class}/"
# original_frames = read_frames(original_video_path)

# Evaluation with the frames
ssim_values = []
original_frames = []
colored_frames = []

def get_frames(path, frame) -> torch.Tensor:
    read_frame = Image.open(os.path.join(path, frame))
    read_frame = read_frame.resize(image_size)
    read_frame = to_tensor(read_frame)//255

    return read_frame

# Loop for all images
for idx, frame in enumerate(list_frame):
    
    original_frames.append(get_frames(original_video_path, frame))

    colored_frames.append(get_frames(colored_video_path, frame))


# List with frames to Torch Tensor
t_original_frmaes = torch.stack(original_frames)
t_colored_frames = torch.stack(colored_frames)

# SSIM values over all frames
ssim_frame = ssim(t_original_frmaes, t_colored_frames)
print(f"SSIM values {ssim_frame}")


# PSNR Values
psnr_frame = psnr(t_original_frmaes, t_colored_frames)
print(f"PSNR values {psnr_frame}")