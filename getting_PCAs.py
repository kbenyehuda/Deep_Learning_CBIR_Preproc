from sklearn.decomposition import PCA
import numpy as np
import pickle
from single_vec_per_pic import *

# base_path = 'D:/Users/Keren/Documents/university/Year 3/deep learning/my_project/'
biggest_SIFT_feats_pickle_path = 'Random1499_SIFT_feats.pickle'
# biggest_SIFT_feats_pickle_path = base_path + 'random_50_SIFT_feats_2.pickle'
n_components = 1499 # if it's too much, you can make it smaller


pickle_in = open(biggest_SIFT_feats_pickle_path,'rb')
all_sift_feats = pickle.load(pickle_in)

pic_names,sift_feats = create_array_of_sift_feats(all_sift_feats)

pca = PCA(n_components=n_components)

pca.fit(sift_feats)
PCA_feats_0 = pca.transform(sift_feats)
norms = np.sqrt(np.sum(np.square(sift_feats),axis=1))
norms = norms[:,np.newaxis]
PCA_feats_0 = np.concatenate((PCA_feats_0,norms),axis=1)
del all_sift_feats

##############################################

SIFT_feats_pickle_path = 'label_1_SIFT_feats.pickle'
pickle_in = open(SIFT_feats_pickle_path,'rb')
all_sift_feats = pickle.load(pickle_in)
pic_names,sift_feats = create_array_of_sift_feats(all_sift_feats)
PCA_feats_1 = pca.transform(sift_feats)
norms = np.sqrt(np.sum(np.square(sift_feats),axis=1))
norms = norms[:,np.newaxis]
PCA_feats_1 = np.concatenate((PCA_feats_1,norms),axis=1)
del all_sift_feats

SIFT_feats_pickle_path = 'label_2_SIFT_feats.pickle'
pickle_in = open(SIFT_feats_pickle_path,'rb')
all_sift_feats = pickle.load(pickle_in)
pic_names,sift_feats = create_array_of_sift_feats(all_sift_feats)
PCA_feats_2 = pca.transform(sift_feats)
norms = np.sqrt(np.sum(np.square(sift_feats),axis=1))
norms = norms[:,np.newaxis]
PCA_feats_2 = np.concatenate((PCA_feats_2,norms),axis=1)
del all_sift_feats

SIFT_feats_pickle_path = 'label_3_SIFT_feats.pickle'
pickle_in = open(SIFT_feats_pickle_path,'rb')
all_sift_feats = pickle.load(pickle_in)
pic_names,sift_feats = create_array_of_sift_feats(all_sift_feats)
PCA_feats_3 = pca.transform(sift_feats)
norms = np.sqrt(np.sum(np.square(sift_feats),axis=1))
norms = norms[:,np.newaxis]
PCA_feats_3 = np.concatenate((PCA_feats_3,norms),axis=1)
del all_sift_feats

############################################
'''
HERE YOU NEED TO CHECK THE SIZE OF EACH AND SEE IF THE AXIS IS REALLY 0 OR IF YOU NEED TO ADD AN AXIS OR SOMETHING
'''

all_PCA_feats = np.concatenate((PCA_feats_0,PCA_feats_1,PCA_feats_2,PCA_feats_3))
dic = {'all_PCA_feats':all_PCA_feats}
with open('all_PCA_feats.pickle', 'wb') as f:
  pickle.dump(dic, f)



