# %% [markdown]
# # **Pose Detection For Weight Lifting and Exercise**

# %% [markdown]
# # SETUP

# %% [markdown]
# ## Import Dependencies

# %%
# !pip install numpy
# !pip install pandas
# !pip install ultralytics
# !pip install matplotlib
# !pip install opencv-python
# !pip install pillow
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install tqdm

# %%
import numpy as np
import pandas as pd
import random

# YOLO
import ultralytics
from ultralytics import YOLO
import ultralytics.utils as plot


# Images
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageFont
from PIL.ImageDraw import Draw

# Torch
import torch
from torch import nn
import torchvision
import torchvision.transforms.v2 as T

# Files
import os
from os.path import join, split
from glob import glob

# Others
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# %%
print(f"Numpy Version: {np.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"OpenCV Version: {cv2.__version__}")

# %% [markdown]
# ## CONFIGURATION

# %%
class CFG:
    path = "./"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42

print(f"DEVICE: {CFG.device}")

# %% [markdown]
# ## REPRODUCIBILITY

# %%
torch.manual_seed(CFG.random_seed) # PyTorch
np.random.seed(CFG.random_seed) # NumPy
random.seed(CFG.random_seed) # Python

# %% [markdown]
# # LOAD MODEL

# %%
model = YOLO('yolov8n-pose.pt', verbose=False) 

# %%
def preprocess_img(image:np.array,
                   transform = T.Resize((640,640))):
    image = image if image.shape[0] == 3 else image.transpose(2,0,1)
    if image.max() > 1.0:
        image = image/255.0
    ch, h, w = image.shape
    image = image.reshape(1, ch, h, w)
    return transform(torch.Tensor(image))


# %% [markdown]
# # DETECTION

# %%
def detect(img):
    if img.shape[0] ==3:
        _,h,w = img.shape
    else:
        h,w,_ = img.shape

    revert_transform = T.Resize((h,w))

    result = model.predict(preprocess_img(img), conf=0.75)[0]


    widths = [int(i[2]) for i in result.boxes.xywh]
    heights = [int(i[3]) for i in result.boxes.xywh]
    areas = [int(widths[i]*heights[i]) for i in range(len(widths))]


    for i,r in enumerate(list(result)):
        if areas[i] == max(areas):
            im_array = r.plot(kpt_line=True, kpt_radius=10)
            im_array = revert_transform(torch.Tensor(im_array.transpose(2,0,1).reshape(1,3,640,640))).cpu().numpy().astype(np.uint8).squeeze().transpose(1,2,0)
            im_array = im_array[:,:,::-1]

            keypoints = r.keypoints.xy

    try:
        return im_array, keypoints[0]
    except:
        return img[:,:,::-1], None

# %%
def scale(x_scale, y_scale, keypoint):
    x = keypoint[0]*x_scale
    y = keypoint[1]*y_scale
    return x.cpu().numpy(),y.cpu().numpy()

# %%
def arm_angle(keypoints, h, w):
    x_scale, y_scale = w/640, h/640
    Lshoulder = scale(x_scale, y_scale, keypoints[5])
    Rshoulder = scale(x_scale, y_scale, keypoints[6])
    Lelbow = scale(x_scale, y_scale, keypoints[7])
    Relbow = scale(x_scale, y_scale, keypoints[8])
    Lhand = scale(x_scale, y_scale, keypoints[9])
    Rhand = scale(x_scale, y_scale, keypoints[10])

    LAngle = np.arctan2(Lhand[1]-Lelbow[1], Lhand[0]-Lelbow[0]) - np.arctan2(Lshoulder[1]-Lelbow[1], Lshoulder[0]-Lelbow[0])
    LAngle = np.abs(LAngle*180.0/np.pi)

    RAngle = np.arctan2(Rhand[1]-Relbow[1], Rhand[0]-Relbow[0]) - np.arctan2(Rshoulder[1]-Relbow[1], Rshoulder[0]-Relbow[0])
    RAngle = np.abs(RAngle*180.0/np.pi)

    return LAngle, RAngle
    
def arm_lift_angle(keypoints, h, w):
    x_scale, y_scale = w/640, h/640
    Lshoulder = scale(x_scale, y_scale, keypoints[5])
    Rshoulder = scale(x_scale, y_scale, keypoints[6])
    Lhand = scale(x_scale, y_scale, keypoints[9])
    Rhand = scale(x_scale, y_scale, keypoints[10])

    Lhip = scale(x_scale, y_scale, keypoints[11])
    Rhip = scale(x_scale, y_scale, keypoints[12])

    LAngle = np.arctan2(Lhand[1]-Lshoulder[1], Lhand[0]-Lshoulder[0]) - np.arctan2(Lhip[1]-Lshoulder[1], Lhip[0]-Lshoulder[0])
    LAngle = np.abs(LAngle*180.0/np.pi)

    RAngle = np.arctan2(Rhand[1]-Rshoulder[1], Rhand[0]-Rshoulder[0]) - np.arctan2(Rhip[1]-Rshoulder[1], Rhip[0]-Rshoulder[0])
    RAngle = np.abs(RAngle*180.0/np.pi)

    return LAngle, RAngle


# %%



