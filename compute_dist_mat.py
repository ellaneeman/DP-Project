"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import numpy as np
# import tensorflow as tf
# import glove_utils
import pickle 
# from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cdist

MAX_VOCAB_SIZE = 50000
print('load emb matrix')
embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
print('loaded')
# print('load missed')
# missed = np.load(('aux_files/missed_embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
# print('before c')
# print(embedding_matrix.shape)
# c_ = -2*(embedding_matrix.T @ embedding_matrix)
# print('after c')
# a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
# print('after a')
# b = a.T
# print('after b')
# dist = a+b+c_
# np.save(('aux_files/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)), dist)
# print('saved file')

# Try an example
with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
print('opened')
src_word = dataset.dict['good']
# neighbours, neighbours_dist = glove_utils.pick_most_similar_words(src_word, dist)
# print('Closest words to `good` are :')
# result_words = [dataset.inv_dict[x] for x in neighbours]
# print(result_words)

print('our')
tra = embedding_matrix.T
# print(src_word)
src_embedding = tra[src_word]
# print(src_embedding)
distances = cdist([src_embedding], tra, "euclidean")[0]
neighbours = np.argsort(distances)[1:11]
result_words = [dataset.inv_dict[x] for x in neighbours]
print(result_words)
