import cv2
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from architectures.swin_unet import Swin_Unet

def read_frames(image_folder,):
    """
    Read all frames of the image_folder, salve it
    in a folder and return.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    if images == []:
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    return images

def frame_2_gray(images) -> list:
    """
    Recive a list of images and return the same list
    but with the images in gray scale.
    """

    gray_frames = []

    for img in images:
        gray_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return gray_frames

def frame_2_video(image_folder, video_name, gray=False, frame_rate=16):
    """
    Get the path with the frames and the name that video must be
    and create and save the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    images = read_frames(image_folder)
        
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    if gray == True:
        video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width,height), 0)
    else:
        video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width,height))

    for image in images:
        temp_image = cv2.imread(os.path.join(image_folder, image), 1)
        
        if gray == True:
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        video.write(temp_image)

    video.release()
    # print("Convertion Done")


################## Losses #####################
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from piq import ssim, SSIMLoss
from piq import VIFLoss
from piq import VSILoss
from torch.nn import KLDivLoss
from architectures_losses.vgg_loss import VGGLoss
from architectures_losses.smooth_loss import Loss
# from pytorch_tools import losses


def valid_loss(loss) -> torch.Tensor:
    """
    Recive a loss and verify if it is a torch.Tensor,
    and return a Tensor if it is a list.
    """
    if type(loss) == list:
        loss = loss[0]

    return loss

def model_losses(losses: list, inputs: list):
    """
    losses: List of loss functions to calculate the loss
    inputs: List of data to calculate the loss.

    return a sum of all losses and List contains all losses individually
    """

    dict_losses = {}

    # train_losses = []

    total_losses = 0

    # populate values of each loss
    for idx, loss in enumerate(losses):
        #loss name
        loss_name = str(type(loss)).split(".")[-1].split("'")[0]

        if len(dict_losses) == 0:
            dict_losses[loss_name] = valid_loss(loss(inputs[0], inputs[1]))
        if len(dict_losses) < 2:
            dict_losses[loss_name] = valid_loss(loss(inputs[3], inputs[1]))
        else:
            dict_losses[loss_name] = valid_loss(loss(inputs[3], inputs[2]))
            
    assert len(dict_losses) == len(losses), "Loss and Output has to have same size"
    # # create variables for each loss
    # for key, value in dict_losses.items():
    #     exec(f"{key}={value}")
    
    # calculate total loss
    for key, value in dict_losses.items():
        
        total_losses += value

    return total_losses, dict_losses

def commet_log_metric(experiment, metric_name: str, metric, step: int, me_type="train") -> None:
    """
    Generic method to create the experiment log
    """
    experiment.log_metric(f"{metric_name}_{me_type}", metric, step=step)


def create_gray_videos(dataset, path_video_save):

    images_paths = f"C:/video_colorization/data/train/{dataset}"
    img_classes = os.listdir(images_paths)

    os.makedirs(path_video_save, exist_ok=True)

    for v_class in img_classes:

    # video_name = "C:/video_colorization/data/videos/gray/sunset_gray.mp4"
        # video_name = f"C:/video_colorization/data/videos/videvo_gray/{v_class}.mp4"
        image_folder = f"C:/video_colorization/data/train/{dataset}/{v_class}"
        
        video_name = f'{path_video_save}{v_class}.mp4'

        frame_2_video(image_folder, video_name, True)

    assert len(img_classes) == len(os.listdir(path_video_save)), "Created videos must be same amout of files that video classes."

    print("Gray videos created")

from architectures.color_model_simple import ColorNetwork

def load_trained_model(model_path, image_size, device):

    #old
    # model = ColorNetwork(in_channel=1, out_channel=128, stride=2, padding=2,img_size=image_size[0]).to(device)

    checkpoint = torch.load(model_path)
    #Get the number of channels in the first layer
    ch_deep = 128
    # ch_deep = checkpoint["inc.double_conv.0.weight"].shape[0]

    # model = ColorNetwork(1, 3, image_size[0], ch_deep).to(device)
    model = Swin_Unet(net_dimension=ch_deep, c_out=3, img_size=image_size).to(device)
    model.load_state_dict(checkpoint)
    model.to(device)

    return model

def generate_paper_colored_samples(
    # dataset = "DAVIS",
    root_path_images = f"C:/video_colorization/Vit-autoencoder/temp_result/DAVIS",
    frame_number = "00042",
    root_path_destiny = "images_to_paper",
    video_name = "rallye.mp4"):

    os.makedirs(root_path_destiny, exist_ok=True)

    import shutil
    list_models = os.listdir(root_path_images)

    for model in list_models:
        origin_path = f"{root_path_images}/{model}/{video_name}/{frame_number}.jpg"

        destiny_path = f"{root_path_destiny}/{video_name}"
        os.makedirs(destiny_path, exist_ok=True)

        shutil.copy(origin_path, f"{destiny_path}/{model}_{frame_number}.jpg")

# Gerenate grayscale videos of all videos in DAVIS_val

# for class_video in os.listdir("C:/video_colorization/data/train/DAVIS_val"):
#     image_folder = f"C:/video_colorization/data/train/DAVIS_val/{class_video}"
#     path_video_save = "C:/video_colorization/data/videos/gray/"
#     video_name = f'{path_video_save}video_{class_video}.mp4'

#     frame_2_video(image_folder, video_name, True)


# image_folder = "C:/video_colorization/Vit-autoencoder/temp_result/20221207_111837"

# class_video = "breakdance"

# image_folder = f"C:/video_colorization/data/train/DAVIS_val/{class_video}"
# path_video_save = "C:/video_colorization/data/videos/gray/"
# video_name = f'{path_video_save}video_{class_video}.mp4'

# frame_2_video(image_folder, video_name, True)

def get_model_time():
    from datetime import datetime
    #to create the timestamp
    dt = datetime.now()
    # dt_str = datetime.timestamp(dt)

    dt_str = str(dt).replace(':','.')
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    return dt_str

def save_losses(dic_losses, filename="losses_network_v1"):
    os.makedirs("losses", exist_ok=True)
    fout = f"{filename}.csv"
    fo = open(fout, "w")

    for k, v in dic_losses.items():
        fo.write(str(k) + "," +str(float(v.cpu().numpy())) + '\n')

    fo.close()

# torch tensor to image
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, x.shape[2], x.shape[2])

    return x  

def create_samples(data, constrative=False):
    """
    img: Image with RGB colors (ground truth)
    img_gray: Grayscale version of the img (this) variable will be used to be colorized
    img_color: the image with color that bt used as example (first at the scene)
    """

    # Test if the pos_color must be returned
    if len(data) == 4:
        img, img_color, next_frame, random_frame = data
        if isinstance(img, list):
            img, img_color, next_frame, random_frame = img[0], img_color[0], next_frame[0], random_frame[0]
    else:
        img, img_color, next_frame = data

        if isinstance(img, list):
            img, img_color, next_frame = img[0], img_color[0], next_frame[0]

    img_gray = transforms.Grayscale(num_output_channels=3)(img)
    gray_next_frame = transforms.Grayscale(num_output_channels=3)(next_frame)
    # img_gray = img[:,:1,:,:]

    if constrative:
        return img, img_gray, img_color, next_frame
    else:
        return img, img_gray, img_color, gray_next_frame


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def scale_0_and_1(tensor):
    """
    Recives a tensor and return their values between 0 and 1
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor_rescaled = (tensor - tensor_min) / (tensor_max - tensor_min)

    return tensor_rescaled

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def resume(model, filename):
    model.load_state_dict(torch.load(filename))