import os
import numpy as np
from skimage import io, exposure



def make_masks():
    path = 'D:\\学习\\MLMI\\Lung-Segmentation-master\\VGG_UNet\\code\\dataset\\CXR_png\\'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('D:\\学习\\MLMI\\Lung-Segmentation-master\\VGG_UNet\\code\\dataset\\ManualMask\\leftMask\\' + filename[:-4] + '.png')
        right = io.imread('D:\\学习\\MLMI\\Lung-Segmentation-master\\VGG_UNet\\code\\dataset\\ManualMask\\rightMask\\' + filename[:-4] + '.png')
        io.imsave('D:\\学习\\MLMI\\Lung-Segmentation-master\\VGG_UNet\\code\\dataset\\ManualMask\\Mask\\' + filename[:-4] + '.png', np.clip(left + right, 0, 255))
        print ('Mask', i, filename)

make_masks()
    
