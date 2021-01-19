import numpy as np
import cv2


def return_label(num_pic):
    if num_pic < 709:
        return 1
    elif num_pic < 2135:
        return 2
    else:
        return 3

def load_image(img_path):
    pic = cv2.imread(img_path)
    pic = np.squeeze(pic[:, :, 0] / np.max(pic))
    return pic

def dist_in_featspace(feat_vec,pic_dic,num_best = 1):
    pic_dic_keys = list(pic_dic.keys())
    min_dists = np.inf*np.ones((num_best,))
    best_keys = list(np.ones((num_best,)))
    for i in range(len(pic_dic_keys)):
        cur_feat_vec = pic_dic[pic_dic_keys[i]]
        dist = np.sum(np.square(np.array(feat_vec)-np.array(cur_feat_vec)))
        max_min_dist = np.max(min_dists)
        if dist<max_min_dist:
            ind = np.where(min_dists==max_min_dist)
            if len(ind[0])>1:
                min_dists[ind[0][0]] = dist
                best_keys[int(ind[0][0])] = pic_dic_keys[i]
            else:
                min_dists[ind[0]] = dist
                best_keys[int(ind[0])] = pic_dic_keys[i]

    min_dists_unique = np.unique(min_dists)
    new_best_keys = []
    for i in range(len(best_keys)):
        ind = np.where(min_dists==min_dists_unique[i])
        new_best_keys.append(best_keys[ind])

    return new_best_keys


def dist_in_new_featspace(feat_vec,pic_array,pic_names,num_best = 1):
    min_dists = np.inf*np.ones((num_best,))
    best_keys = list(np.ones((num_best,)))
    for i in range(len(pic_array)):
        cur_feat_vec = pic_array[i,:]
        dist = np.sum(np.square(np.array(feat_vec)-np.array(cur_feat_vec)))
        max_min_dist = np.max(min_dists)
        if dist<max_min_dist:
            ind = np.where(min_dists==max_min_dist)
            if len(ind[0])>1:
                min_dists[ind[0][0]] = dist
                best_keys[int(ind[0][0])] = pic_names[i]
            else:
                min_dists[ind[0]] = dist
                best_keys[int(ind[0])] = pic_names[i]
    min_dists_unique = np.unique(min_dists)
    new_best_keys = []
    for i in range(len(best_keys)):
        ind = np.where(min_dists==min_dists_unique[i])
        new_best_keys.append(best_keys[int(ind[0])])


    return new_best_keys
