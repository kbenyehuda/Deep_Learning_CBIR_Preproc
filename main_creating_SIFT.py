import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import prange
from glob import glob
from Sift_funcs import *
from ex_funcs import *
from single_vec_per_pic import *

'''
This main is for creating SIFT features for images
label 1: 1-708, i=0-707
label 2: 709-2134, i=708,2133
label 3: 2135-3064, i=2134,3063
'''

pic_filepaths = glob('DataBase/ProjectDB/DataBaseImage/*')
pic_1 = cv2.imread(pic_filepaths[0])
pic_1 = np.squeeze(pic_1[:,:,0]/np.max(pic_1))

window_size = 100
stride = 20

num_rows,num_cols = np.shape(pic_1)

num_its_in_r = int(np.floor((num_rows-window_size)/stride))
num_its_in_c = int(np.floor((num_cols-window_size)/stride))

del pic_1


num_of_pics_to_proc = 5
distance_matrix = np.zeros((num_of_pics_to_proc,num_of_pics_to_proc))
for base_pic_i in prange(num_of_pics_to_proc):
    try:
        # j = np.random.randint(0,num_of_pics_to_proc)
        if base_pic_i%5==0:
            print('started i ', base_pic_i, ' out of ', num_of_pics_to_proc)
        pic = cv2.imread(pic_filepaths[base_pic_i])
        pic = np.squeeze(pic[:, :, 0] / np.max(pic))
        pic_name = pic_filepaths[base_pic_i][33:-4]
        cur_sift = calc_SIFT_for_pic(pic, pic_name, num_its_in_r, num_its_in_c, stride, window_size)

        for all_pics_i in prange(num_of_pics_to_proc):
            if all_pics_i>base_pic_i:
                try:
                    pic = cv2.imread(pic_filepaths[all_pics_i])
                    pic = np.squeeze(pic[:, :, 0] / np.max(pic))
                    pic_name = pic_filepaths[all_pics_i][33:-4]
                    new_sift = calc_SIFT_for_pic(pic, pic_name, num_its_in_r, num_its_in_c, stride, window_size)
                    dist = np.sqrt(np.sum(np.square(np.array(cur_sift) - np.array(new_sift))))
                    distance_matrix[base_pic_i,all_pics_i] = dist
                except:
                    continue

    except:
        continue

# import pickle
# dic = {'distance_matrix':distance_matrix}
# with open('distance_matrix.pickle', 'wb') as f:
#   pickle.dump(dic, f)
