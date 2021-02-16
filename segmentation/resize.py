import numpy as np
import os
from cv2 import cv2
from skimage import io
from skimage.util import random_noise
from skimage.filters import gaussian
from tqdm import tqdm

dim = (256,256)
image_names  = os.listdir(os.getcwd())
for name in tqdm(image_names):
	img = cv2.imread(os.getcwd()+name)
	resized = cv2.resize(img, dim)
	cv2.imwrite(os.getcwd()+name,resized)

