import cv2
import os

def frame_2_video(image_folder, video_name, frame_rate=16):
    """
    Get the path with the frames and the name that video must be
    and create and save the video.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    if images == []:
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, frame_rate, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    print("Convertion Done")

# image_folder = "C:/video_colorization/Vit-autoencoder/temp_result/20221207_111837"

# image_folder = "C:/video_colorization/data/DAVIS_val/cows"
# video_name = 'video_cows.avi'

# frame_2_video(image_folder, video_name)