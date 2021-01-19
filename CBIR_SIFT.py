import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from Sift_funcs import *
from ex_funcs import *
from single_vec_per_pic import *

'''
This main is for testing similarity
label 1: 1-708
label 2: 709-2134
label 3: 2135-3064
'''


import pickle
pickle_in = open('SIFT_feats_1.pickle','rb')
all_sift_feats = pickle.load(pickle_in)

pic_names,sift_feats_array = create_array_of_sift_feats(all_sift_feats)

base_ind = 6
cur_sift = sift_feats_array[base_ind]
best_keys = dist_in_new_featspace(cur_sift,sift_feats_array,pic_names,5)

pic_filepaths = glob('DataBase/ProjectDB/DataBaseImage/*')
base_img_filepath = pic_filepaths[base_ind][:33]
cur_pic = load_image(base_img_filepath+pic_names[base_ind]+'.png')

plt.figure()
plt.subplot(1,len(best_keys)+1,1)
plt.imshow(cur_pic,cmap='gray')
plt.title(pic_names[base_ind]+', label: '+str(return_label(pic_names[base_ind])))
plt.xticks([])
plt.yticks([])

for i in range(len(best_keys)):
    similar_pic = load_image(base_img_filepath+best_keys[i]+'.png')
    plt.subplot(1,len(best_keys)+1,i+2)
    plt.imshow(similar_pic,cmap='gray')
    plt.title(best_keys[i]+', label: '+str(return_label(best_keys[i])))
    plt.xticks([])
    plt.yticks([])
    plt.show()

