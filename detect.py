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
localization = YOLO('yolov8n.pt')

# %%
model = YOLO('yolov8n-pose.pt') 


# %
# %%
def preprocess_img(image:np.array,
                   transform = T.Resize((640,640))):
    image = image if image.shape[0] == 3 else image.transpose(2,0,1)
    if image.max() > 1.0:
        image = image/255.0
    ch, h, w = image.shape
    image = image.reshape(1, ch, h, w)
    return transform(torch.Tensor(image))


# %%
def detect(img):
    if img.shape[0] ==3:
        _,h,w = img.shape
    else:
        h,w,_ = img.shape

    print(h,w)
    revert_transform = T.Resize((h,w))

    result = model(preprocess_img(img))[0]

    confidence = [float(i.boxes.conf) for i in result]
    for r in list(result):
        if r.boxes.conf == max(confidence):
            im_array = r.plot(kpt_line=True, kpt_radius=10)
            im_array = revert_transform(torch.Tensor(im_array.transpose(2,0,1).reshape(1,3,640,640))).cpu().numpy().astype(np.uint8).squeeze().transpose(1,2,0)
            im_array = im_array[:,:,::-1]
    return im_array

