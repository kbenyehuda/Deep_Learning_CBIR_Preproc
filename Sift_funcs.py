import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import prange
from single_vec_per_pic import *

def calc_SIFT_for_pic(pic,pic_name,num_its_in_r,num_its_in_c,stride,window_size):
    all_sift_feats_for_pic = {}
    try:
        for r_i in prange(num_its_in_r):
            for c_i in prange(num_its_in_c):
                cur_window = pic[r_i*stride:r_i*stride+window_size,c_i*stride:c_i*stride+window_size]
                # cv2.imwrite('cur_superpixels/'+str(r_i)+'_'+str(c_i)+'.png',255*np.squeeze(cur_window/np.max(cur_window)))
                sift_feats = SIFT_1(cur_window,size_of_sub_sub_pixel=9)
                all_sift_feats_for_pic[r_i,c_i]=sift_feats
        # all_sift_feats[pic_filepaths[j][33:-4]]=all_sift_feats_for_pic
        sift_feats = create_array_of_sift_feats_per_pic(all_sift_feats_for_pic)
    except:
        print('unable to complete for ',pic_name)
        return False

    return sift_feats


def calc_orig_hist_in_window(window,num_bins):
    range_of_bin = int(np.round(255/num_bins))
    val_hist = {}
    for i in range(num_bins):
        val_hist[range_of_bin*i] = np.sum(
            np.logical_and(
                window>range_of_bin*i,window<range_of_bin*(i+1)
            )
        )

    return val_hist


def calc_SIFT_in_window(window,num_bins,return_major_dir = True,return_hist=False,major_dir = None,show_Mag = False,
                        show_Dir = False):
    # calculating the direction

    eps = 10**(-5)

    window_moved_in_x_pos = np.zeros_like(window)
    window_moved_in_x_pos[1:,:] = window[:-1,:]

    window_moved_in_x_neg = np.zeros_like(window)
    window_moved_in_x_neg[:-1, :] = window[1:, :]

    window_moved_in_y_pos = np.zeros_like(window)
    window_moved_in_y_pos[:, 1:] = window[:, :-1]

    window_moved_in_y_neg = np.zeros_like(window)
    window_moved_in_y_neg[:, :-1] = window[:, 1:]

    X_grad = window_moved_in_x_pos - window_moved_in_x_neg
    X_grad = X_grad[1:-1,1:-1]

    Y_grad = window_moved_in_y_pos - window_moved_in_y_neg
    Y_grad = Y_grad[1:-1, 1:-1]

    Mag = np.sqrt(
        np.square(X_grad) +
        np.square(Y_grad)
    )

    if show_Mag:
        plt.imshow(Mag.astype(np.uint8))
        plt.show()

    # Dir = np.arctan2(
    #     (window_moved_in_y_pos - window_moved_in_y_neg + eps),
    #     (window_moved_in_x_pos - window_moved_in_x_neg)
    # )
    Dir = np.arctan(
        (Y_grad)/
        (X_grad + eps)
    )
    Dir_deg = Dir*180/(np.pi)

    if major_dir:
        Dir_deg = Dir_deg - major_dir

    if show_Dir:
        plt.imshow(Dir_deg)
        plt.show()

    val = 360/num_bins
    Dir_quantized = np.round(Dir_deg/val)*val
    all_quantized_directions = np.unique(Dir_quantized)

    hist = {}
    for i in range(len(all_quantized_directions)):
        inds = np.where(Dir_quantized==all_quantized_directions[i])
        Sum = np.sum(Mag[inds])
        if return_hist:
            hist[all_quantized_directions[i]] = Sum
        else:
            hist[Sum] = all_quantized_directions[i]

    if return_hist:
        all_dic_vals = list(hist.keys())
        all_vals = np.linspace(-180,180,9)
        for i in range(len(all_vals)):
            val = all_vals[i]
            if val not in all_dic_vals:
                hist[val]=0
        return hist

    if return_major_dir:
        try:
            major_direction = hist[np.max(list(hist.keys()))]
        except ValueError:
            major_direction = 0
        return major_direction

def SIFT_2(window,num_of_orig_val_bins = 10,show_Mag=False,size_of_sub_sub_pixel=5):
    major_dir = calc_SIFT_in_window(window,num_bins = 36,return_major_dir=True)

    # the new way
    num_of_subpixels = int(np.ceil(np.shape(window)[0]/2)/size_of_sub_sub_pixel)
    r_c = int(np.round(np.shape(window)[0] / 2))
    c_c = int(np.round(np.shape(window)[1] / 2))

    orig_vals_hists = []
    subpixel_hists = []

    for i in range(num_of_subpixels):
        if i==0:
            num_of_subsubs = 1
            c_cur = r_c
            r_cur = c_c
            sub_window = window[r_cur:r_cur + size_of_sub_sub_pixel, c_cur:c_cur + size_of_sub_sub_pixel]
            cur_hist=[calc_SIFT_in_window(sub_window, num_bins=8, return_major_dir=False, return_hist=True,
                                                major_dir=major_dir, show_Mag=show_Mag)]
            orig_vals_hist = [calc_orig_hist_in_window(sub_window, num_of_orig_val_bins)]

        else:
            num_of_subsubs = i*8
            num_in_r = i*2 + 1

            st_pt_r = r_c - i*size_of_sub_sub_pixel
            st_pt_c = c_c - i*size_of_sub_sub_pixel

            cur_hist = []
            orig_vals_hist = []
            for pt in range(4):
                if pt == 0:
                    # print('pt: ',pt)
                    r_cur = st_pt_r
                    for j in range(num_in_r - 1):
                        c_cur = st_pt_c + size_of_sub_sub_pixel*j
                        # print('r: ',r_cur,', c: ',c_cur)
                        sub_window = window[r_cur:r_cur + size_of_sub_sub_pixel,c_cur:c_cur + size_of_sub_sub_pixel]
                        cur_hist.append(calc_SIFT_in_window(sub_window, num_bins=8, return_major_dir=False, return_hist=True,
                                                            major_dir = major_dir,show_Mag = show_Mag))
                        orig_vals_hist.append(calc_orig_hist_in_window(sub_window,num_of_orig_val_bins))
                if pt == 1:
                    # print('pt: ',pt)
                    c_cur = st_pt_c + (num_in_r -1)*size_of_sub_sub_pixel
                    for j in range(num_in_r - 1):
                        r_cur = st_pt_r + size_of_sub_sub_pixel*j
                        # print('r: ', r_cur, ', c: ', c_cur)
                        sub_window = window[r_cur:r_cur + size_of_sub_sub_pixel,c_cur:c_cur + size_of_sub_sub_pixel]
                        cur_hist.append(calc_SIFT_in_window(sub_window, num_bins=8, return_major_dir=False, return_hist=True,
                                                            major_dir = major_dir,show_Mag = show_Mag))
                        orig_vals_hist.append(calc_orig_hist_in_window(sub_window, num_of_orig_val_bins))

                if pt == 2:
                    # print('pt: ',pt)
                    r_cur = st_pt_r + (num_in_r - 1)*size_of_sub_sub_pixel
                    for j in range(num_in_r - 1):
                        c_cur = st_pt_c + (num_in_r -1 -j)*size_of_sub_sub_pixel
                        # print('r: ', r_cur, ', c: ', c_cur)
                        sub_window = window[r_cur:r_cur + size_of_sub_sub_pixel,c_cur:c_cur + size_of_sub_sub_pixel]
                        cur_hist.append(calc_SIFT_in_window(sub_window, num_bins=8, return_major_dir=False, return_hist=True,
                                                            major_dir = major_dir,show_Mag = show_Mag))
                        orig_vals_hist.append(calc_orig_hist_in_window(sub_window, num_of_orig_val_bins))

                if pt == 3:
                    # print('pt: ',pt)
                    c_cur = st_pt_c
                    for j in range(num_in_r - 1):
                        r_cur = st_pt_r + (num_in_r -1 -j)*size_of_sub_sub_pixel
                        # print('r: ', r_cur, ', c: ', c_cur)
                        sub_window = window[r_cur :r_cur + size_of_sub_sub_pixel,c_cur:c_cur + size_of_sub_sub_pixel]
                        cur_hist.append(calc_SIFT_in_window(sub_window, num_bins=8, return_major_dir=False, return_hist=True,
                                                            major_dir = major_dir,show_Mag = show_Mag))
                        orig_vals_hist.append(calc_orig_hist_in_window(sub_window, num_of_orig_val_bins))

        subpixel_hists.append(cur_hist)
        orig_vals_hists.append(orig_vals_hist)

    sift_feats = []
    for i in range(len(subpixel_hists)):
        new_hist = {}
        all_angles = np.unique(list(subpixel_hists[i][0].keys()))
        for angle_key in all_angles:
            angle_value = 0
            for subsub_i in range(len(subpixel_hists[i])):
                angle_value += subpixel_hists[i][subsub_i][angle_key]
            new_hist[angle_key] = angle_value
        hist_vals = np.array(list(new_hist.values()))
        overall_sum = np.sum(np.array(hist_vals)) + 10**(-5)
        hist_vals = list(hist_vals / overall_sum)

        orig_hist = {}
        all_orig_vals = np.unique(list(orig_vals_hists[i][0].keys()))
        for orig_val in all_orig_vals:
            orig_sum = 0
            for subsub_i in range(len(orig_vals_hists[i])):
                orig_sum += orig_vals_hists[i][subsub_i][orig_val]
            orig_hist[orig_val] = orig_sum
        orig_hist_vals = np.array(list(orig_hist.values()))
        overall_orig_sum = np.sum(np.array(orig_hist_vals)) + 10**(-5)
        final_orig_vals = list(orig_hist_vals/overall_orig_sum)

        sift_feats += list(hist_vals)+list(final_orig_vals)
    return sift_feats


def SIFT_1(window,num_of_orig_val_bins = 10,show_Mag=False,size_of_sub_sub_pixel=5):
    major_dir = calc_SIFT_in_window(window,num_bins = 36,return_major_dir=True)
    subpixel_hists = []

    ## the old way of 4x4 subpixels
    orig_vals_hists = []
    r_sub_window = int(np.shape(window)[0] / size_of_sub_sub_pixel)
    c_sub_window = int(np.shape(window)[1] / size_of_sub_sub_pixel)
    for r_i in range(4):
        for c_i in range(4):
            sub_window = window[r_i*r_sub_window:(r_i+1)*r_sub_window,c_i*c_sub_window:(c_i+1)*c_sub_window]
            subpixel_hists.append(calc_SIFT_in_window(sub_window,num_bins = 8,return_major_dir=False,return_hist=True,
                                                 major_dir = major_dir,show_Mag = show_Mag))
            orig_vals_hists.append(calc_orig_hist_in_window(sub_window, num_of_orig_val_bins))

    sift_feats = []
    for i in range(len(subpixel_hists)):
        new_hist = {}
        all_angles = np.unique(list(subpixel_hists[i].keys()))
        for angle_key in all_angles:
            angle_value = subpixel_hists[i][angle_key]
            new_hist[angle_key] = angle_value
        hist_vals = np.array(list(new_hist.values()))
        overall_sum = np.sum(np.array(hist_vals)) + 10**(-5)
        hist_vals = list(hist_vals / overall_sum)

        orig_hist = {}
        all_orig_vals = np.unique(list(orig_vals_hists[i].keys()))
        for orig_val in all_orig_vals:
            orig_sum = orig_vals_hists[i][orig_val]
            orig_hist[orig_val] = orig_sum
        orig_hist_vals = np.array(list(orig_hist.values()))
        overall_orig_sum = np.sum(np.array(orig_hist_vals)) + 10**(-5)
        final_orig_vals = list(orig_hist_vals/overall_orig_sum)

        sift_feats += list(hist_vals)+list(final_orig_vals)
    return sift_feats