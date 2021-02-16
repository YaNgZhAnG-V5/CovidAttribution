import numpy as np
import os
from cv2 import cv2
from skimage import io
from skimage.util import random_noise
from skimage.filters import gaussian
from tqdm import tqdm
'''
image_names  = os.listdir('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/train/CXR_png/')

for name in tqdm(image_names):
    img = io.imread('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/train/CXR_png/'+name)
    mask = io.imread('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/train/Mask/'+name)

    noise_image_01 = random_noise(img, mode='gaussian', seed=None, clip=True, var = 0.01)
    noise_image_01 = noise_image_01*255.
    noise_image_01 = noise_image_01.astype('uint8')
    io.imsave('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/train/CXR_png/'+name[:-4]+'_01.png', noise_image_01)
    io.imsave('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/train/Mask/'+name[:-4]+'_01.png', mask)

'''
image_names  = os.listdir('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/test/CXR_png/')


for name in tqdm(image_names): 
	img = cv2.imread('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/test/CXR_png/'+name,0)
	mask = cv2.imread('D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/test/Mask/'+name,0)
 
# copy image to display all 4 variations
	#horizontal_img = img.copy()
	#horizontal_mask = mask.copy()

 
# flip img horizontally, vertically,
# and both axes with flip()
	horizontal_img = cv2.flip(img, 1 )
	horizontal_mask = cv2.flip(mask, 1)

 
# display the images on screen with imshow()
	cv2.imshow( 'D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/test/CXR_png/'+name[:-4]+'_flip.png', horizontal_img )
	cv2.imshow( 'D:/学习/MLMI/Lung-Segmentation-master/VGG_UNet/code/dataset/test/Mask/'+name[:-4]+'_flip.png', horizontal_mask )
