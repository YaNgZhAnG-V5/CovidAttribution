from PIL import Image
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import os
import json
import time

#uncomplete
def calculate_ior_iobb(cam, bbox):
    '''
    calculate ior and iobb

    args:
        cam: path of generated cam images in .jpg format
        bbox: [x,y,w,h]
    '''
    cam_jpg = Image.open(cam)
    cam = np.array(cam_jpg)
    bbox_mask = np.zeros(cam.shape, dtype=bool)
    bbox_mask[bbox[0]: bbox[0]+bbox[2], bbox[1]: bbox[1]+bbox[3]] = True
    activation_mask = np.logical_or(cam >= 180 , cam <= 60)
    label_im, _nb_labels = ndimage.label(activation_mask)
    object_slices = ndimage.find_objects(label_im)
    detected_patchs = []
    for object_slice in object_slices:
        y_slice = object_slice[0]
        x_slice = object_slice[1]
        xy_corner = (x_slice.start, y_slice.start)
        x_length = x_slice.stop - x_slice.start
        y_length = y_slice.stop - y_slice.start
        detected_patch = patches.Rectangle(xy_corner, x_length, y_length, linewidth=2, edgecolor='m',
                                           facecolor='none', zorder=2)
        detected_patchs.append(detected_patch)
    object_masks = []
    for object_slice in object_slices:
        object_mask = np.zeros(label_im.shape, dtype=bool)
        object_mask[object_slice[0], object_slice[1]] = True
        object_masks.append(object_mask)
    object_masks = np.array(object_masks)
    object_masks_union = np.logical_or.reduce(object_masks)

    def compute_ior(activated_mask, gt_mask):
        intersection_mask = np.logical_and(activated_mask, gt_mask)
        detected_region_area = np.sum(activated_mask)
        intersection_area = np.sum(intersection_mask)
        ior = intersection_area / detected_region_area
        return ior

    ior = compute_ior(activation_mask, bbox_mask)
    #print('ior:')
    #print(ior)
    iobb = compute_ior(object_masks_union, bbox_mask)
    #print('iobb:')
    #print(iobb)

#complete
def load_cam(cam_path, format='png'):
    '''
    return a list of all files(cams) from cam_path
    '''
    cam_list_png = []
    cam_list_jpg = os.listdir(cam_path)
    for cam in cam_list_jpg:
         cam_list_png.append(str(cam).replace('.jpg','_mask_post.png'))
    if format == 'jpg':
        return cam_list_jpg
    elif format == 'png':
        return cam_list_png

#complete
def load_bbox(csv_path, img, score):
    '''
    img: name of mask img, str
    score: determine which score(s) we are interested in, dtype = list
    return the bbox list of this img
    '''
    df = pd.read_csv(csv_path)
    bbox_list = []

    idx = ['0','1','2','3','4','5']
    for _idx in idx:
        score_bbox_init = df.loc[ df['filename'] == img , _idx]

        if type(score_bbox_init.iloc[0]) is str:                #
            score_bbox_in_str = score_bbox_init.iloc[0]         # these three lines check whether the index correspond to empty cell.
            if type(len(score_bbox_in_str)) is int:             #
                score_bbox_in_list = json.loads(score_bbox_in_str)
                if score_bbox_in_list[0] in score:              #TODO: solve the 0,1 problem in covid_01_2_3_maps
                    bb = score_bbox_in_list[-4:]
                    bbox_list.append(bb)

    return bbox_list

#uncomplete
def iter_dir(method, root_path='F:/ior_iobb/'):
    '''
    return the working dir and the score of interest, both in list
    '''
    rp = os.listdir(root_path)
    dir_list = []
    sub_dir_list = ['excitation_backprop/','grad_cam/','gradient/','ib/','integrated_gradients/','reverse_ib/']

    for ele in rp:
        if os.path.isdir(ele):  
            dir_list.append(ele)

    for ele in dir_list:
        if ele == 'covid_01_2_3_maps' and method == 'covid_01_2_3_maps':
            score_of_interest = [0,1,2,3]
            working_dir = []
            for name in sub_dir_list:
                subsub = os.listdir(os.path.join(root_path+'covid_01_2_3_maps/'+name))
                for subdir_of_subdir in subsub:
                    working_dir.append(os.path.join(root_path+'covid_01_2_3_maps/'+name+subdir_of_subdir+'/'))
            return working_dir, score_of_interest            
        elif ele == 'covid_1_2_3_maps' and method == 'covid_1_2_3_maps':
            score_of_interest = [1,2,3]
            working_dir = []
            for name in sub_dir_list:
                subsub = os.listdir(os.path.join(root_path+'covid_1_2_3_maps/'+name))
                for subdir_of_subdir in subsub:
                    working_dir.append(os.path.join(root_path+'covid_1_2_3_maps/'+name+subdir_of_subdir+'/'))
            return working_dir, score_of_interest
        elif ele == 'covid_2_3_maps' and method == 'covid_2_3_maps':
            score_of_interest = [2,3]
            working_dir = []
            for name in sub_dir_list:
                subsub = os.listdir(os.path.join(root_path+'covid_2_3_maps/'+name))
                for subdir_of_subdir in subsub:
                    working_dir.append(os.path.join(root_path+'covid_2_3_maps/'+name+subdir_of_subdir+'/'))
            return working_dir, score_of_interest
        elif ele == 'covid_binary_maps' and method == 'covid_binary_maps':
            score_of_interest = []   #what to put here?
            working_dir = []
            for name in sub_dir_list:
                subsub = os.listdir(os.path.join(root_path+'covid_binary_maps/'+name))
                for subdir_of_subdir in subsub:
                    working_dir.append(os.path.join(root_path+'covid_binary_maps/'+name+subdir_of_subdir+'/'))
            return working_dir, score_of_interest


def exec_this(csv_path='F:/ior_iobb/bbox_final.csv'):
    method = ['covid_01_2_3_maps','covid_1_2_3_maps','covid_2_3_maps','covid_binary_maps']
    for mt in method:
        working_dir, score_of_interest = iter_dir(mt)     
        for wd in working_dir:
            cam_list = load_cam(wd)
            for cam in cam_list:
                bbox_list = load_bbox(csv_path, cam, score_of_interest)
                #print(bbox_list)
                for bb in bbox_list:
                    cam_location = os.path.join(wd,cam).replace('_mask_post.png','.jpg')
                    #print(cam_location)
                    calculate_ior_iobb(cam_location,bb)
                    #print(bb)

start_time1 = time.clock()
exec_this()
end_time1 = time.clock()
time1 = end_time1-start_time1
print('\nRunning time1:', time1)





