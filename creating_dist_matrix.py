from single_vec_per_pic import *
import pickle



def choose_cur_pickle(ind,cur_pickle_name,pic_names,sift_feats_array):
    if ind<1001:
        if cur_pickle_name != 1:
            del pic_names
            del sift_feats_array
            pickle_in = open('Pickles/SIFT_feats_1.pickle', 'rb')
            all_sift_feats = pickle.load(pickle_in)
            pic_names, sift_feats_array = create_array_of_sift_feats(all_sift_feats)
            del pickle_in
            del all_sift_feats
            return pic_names,sift_feats_array,1
    elif ind<2001:
        if cur_pickle_name !=2:
            del pic_names
            del sift_feats_array
            pickle_in = open('Pickles/SIFT_feats_2.pickle', 'rb')
            all_sift_feats = pickle.load(pickle_in)
            pic_names, sift_feats_array = create_array_of_sift_feats(all_sift_feats)
            del pickle_in
            del all_sift_feats
            return pic_names,sift_feats_array,2
    else:
        if cur_pickle_name !=3:
            del pic_names
            del sift_feats_array
            pickle_in = open('Pickles/SIFT_feats_3.pickle', 'rb')
            all_sift_feats = pickle.load(pickle_in)
            pic_names, sift_feats_array = create_array_of_sift_feats(all_sift_feats)
            del pickle_in
            del all_sift_feats
            return pic_names,sift_feats_array,3
    return pic_names,sift_feats_array,cur_pickle_name


pickle_in = open('Pickles/SIFT_feats_1.pickle', 'rb')
all_sift_feats = pickle.load(pickle_in)
pic_names, sift_feats_array = create_array_of_sift_feats(all_sift_feats)
cur_pickle_name = 1
del all_sift_feats
del pickle_in


dists_array = np.zeros((3065,3065))
for base_pic_i in range(len(3064)):
    pic_names,sift_feats_array,cur_pickle_name = choose_cur_pickle(base_pic_i,cur_pickle_name,pic_names,sift_feats_array)
    cur_sift = sift_feats_array[base_pic_i]
    for all_pics_i in range(len(pic_names)):
        pic_names, sift_feats_array, cur_pickle_name = choose_cur_pickle(all_pics_i, cur_pickle_name, pic_names,
                                                                         sift_feats_array)
        if cur_pickle_name == 2:
            cur_feat_vec = sift_feats_array[all_pics_i-1000, :]
        elif cur_pickle_name == 3:
            cur_feat_vec = sift_feats_array[all_pics_i - 2000, :]
        else:
            cur_feat_vec = sift_feats_array[all_pics_i, :]
        dist = np.sqrt(np.sum(np.square(np.array(cur_sift) - np.array(cur_feat_vec))))
        dists_array[base_pic_i,all_pics_i] = dist

dic={'dist_matrix':dists_array}
with open('dists_matrix.pickle', 'wb') as f:
  pickle.dump(dic, f)


