"""
Use the images genereated by the trained model and
compare with the grownd truth.

The results are from SSIM and PSNR values.
"""

from torchvision import transforms
to_tensor = transforms.PILToTensor()

from piq import ssim, psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import DAVIS_dataset as ld
dataLoader = ld.ReadData()
from PIL import Image
import os
from tqdm import tqdm
from utils import *

image_size = (128, 128)
device = "cuda"
# video_class = "parkour"
str_dt = "20221223_160049"

dataset = "DAVIS_val"
# dataset = "videvo"

images_paths = f"C:/video_colorization/data/train/{dataset}"

img_classes = os.listdir(images_paths)

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size[0], image_size[1])
    return x

def get_frames(path, frame) -> torch.Tensor:
    read_frame = Image.open(os.path.join(path, frame))
    read_frame = read_frame.resize(image_size)
    read_frame = to_tensor(read_frame)//255

    return read_frame

root_results = "temp_result"
list_results_folders = os.listdir(root_results)

pbar = tqdm(list_results_folders)
for str_dt in pbar:
    pbar.set_description(f"Processing: {str_dt}")

    # Read frames colored by the model
    # colored_video_path = f"temp_result/{str_dt}/"
    colored_video_path = f"{root_results}/{str_dt}/"

    dict_metrics = {}

    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)

    pbar = tqdm(img_classes)
    for v_class in pbar:
        pbar.set_description(f"Processing: {v_class}")
        try:
            list_frame = read_frames(f"{colored_video_path}{v_class}.mp4")

            # Read the grownd truth images
            original_video_path = f"{images_paths}/{v_class}/"
            # original_frames = read_frames(original_video_path)

            # Evaluation with the frames
            ssim_values = []
            original_frames = []
            colored_frames = []

            # Loop for all images
            for idx, frame in enumerate(list_frame):
                
                original_frames.append(get_frames(original_video_path, frame))

                colored_frames.append(get_frames(f"{colored_video_path}{v_class}.mp4", frame))

            # List with frames to Torch Tensor
            t_original_frmaes = torch.stack(original_frames)
            t_colored_frames = torch.stack(colored_frames)

            # SSIM values over all frames
            ssim_frame = ssim(t_original_frmaes, t_colored_frames)

            # PSNR Values
            psnr_frame = psnr(t_original_frmaes, t_colored_frames)

            # LPISP values
            lpips_frame = lpips(t_original_frmaes, t_colored_frames).item()

            dict_metrics[v_class] = [float(ssim_frame), float(psnr_frame), float(lpips_frame)]
            # dict_metrics[f"{v_class}_ssim"] = ssim_frame
            # dict_metrics[f"{v_class}_psnr"] = psnr_frame

            # print(f"SSIM values {ssim_frame}")

            # print(f"PSNR values {psnr_frame}")
        except FileNotFoundError:
            print(f"Video not found: {v_class} in: {colored_video_path}{v_class}.mp4")
            pass


    metric_path = f"models_metrics/{str_dt}/"
    os.makedirs(metric_path, exist_ok=True)

    import pandas as pd
    df_metrics = pd.DataFrame.from_dict(dict_metrics)
    df_metrics = df_metrics.T
    df_metrics.columns = ["SSIM", "PSNR", "LPISP"]
    df_metrics.to_csv(f"{metric_path}model_metrics.csv")