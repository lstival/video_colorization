import cv2
import os

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
    print("Convertion Done")

# image_folder = "C:/video_colorization/Vit-autoencoder/temp_result/20221207_111837"

# image_folder = "C:/video_colorization/data/train/DAVIS_val/cows"
# video_name = 'video_cows.mp4'

# frame_2_video(image_folder, video_name, True)
