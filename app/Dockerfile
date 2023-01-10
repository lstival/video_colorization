FROM rayproject/ray:1.10.0-py38

## Working directory
WORKDIR /app
COPY requirements.txt ./
ENV CUDA_HOME=/usr/local/cuda

## Install Requirements
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install -r requirements.txt

## Copy script
# COPY ./app.py ./

## Start
CMD ["python3", "./app.py"]


# FROM python:3.10-slim-buster
# # FROM rocker/tidyverse:3.6.1
# # ENV CUDA_HOME=/usr/local/cuda

# WORKDIR /app

# RUN apt-get install git cmake
# RUN pip install --upgrade pip

# COPY requirements.txt requirements.txt

# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# RUN pip install -r requirements.txt

# RUN pip install pytorch_tools==0.1.9
# RUN pip install torch-directml
# RUN pip install comet_ml==3.31.21
# RUN pip install denoising_diffusion_pytorch==1.0.2
# RUN pip install kornia==0.6.9
# RUN pip install matplotlib==3.5.2
# RUN pip install numpy==1.21.5
# RUN pip install opencv_python==4.6.0.66
# RUN pip install pandas==1.4.4
# RUN pip install Pillow==9.4.0
# RUN pip install piq==0.7.0
# RUN pip install pyautogui==0.9.53
# RUN pip install # torch==1.13.1+cu116
# RUN pip install torchmetrics==0.11.0
# RUN pip install torchvision==0.14.1+cu116
# RUN pip install tqdm==4.64.1
# RUN pip install vit_pytorch==0.40.2

# Set up time zone.
# ENV TZ=UTC
# RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# # Install system libraries required by OpenCV.
# RUN sudo apt-get update \
#  && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
#  && sudo rm -rf /var/lib/apt/lists/*

# # Install OpenCV from PyPI.
# # RUN pip install opencv-python==4.5.1.48
# RUN pip install -r requirements.txt