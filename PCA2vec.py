# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:38:41 2020

@author: kbara
"""
import csv

with open('output.csv', newline='') as f:
    reader = csv.reader(f)
    res = list(map(tuple, reader))


sift_feats = all_sift_feats['all_PCA_feats']
dict_of_sift_feats = {}
for i in range(1499):
    cur_pic = sift_feats[i]
    cur_pic_name = data[i]
    dict_of_sift_feats[cur_pic_name[0]] = cur_pic
    
import pickle
with open('PCA_1499.pickle', 'wb') as f:
  pickle.dump(dict_of_sift_feats, f)