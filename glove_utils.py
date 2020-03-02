"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

import numpy as np
import pickle
from scipy.spatial.distance import cdist

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        row = line.strip().split(' ')
        word = row[0]
        #print(word)
        embedding = np.array([float(val) for val in row[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def save_glove_to_pickle(glove_model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(glove_model, f)
        
def load_glove_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def create_embeddings_matrix(glove_model, dictionary, full_dictionary, d=300):
    MAX_VOCAB_SIZE = len(dictionary)
    # Matrix size is 300
    embedding_matrix = np.zeros(shape=((d, MAX_VOCAB_SIZE+1)))
    cnt  = 0
    unfound = []
    
    for w, i in dictionary.items():
        if not w in glove_model:
            cnt += 1
            #if cnt < 10:
            # embedding_matrix[:,i] = glove_model['UNK']
            unfound.append(i)
        else:
            embedding_matrix[:, i] = glove_model[w]
    print('Number of not found words = ', cnt)
    return embedding_matrix, unfound


def pick_most_similar_words(src_word_idx, embeddings, dist_metric="euclidean", ret_count=10, threshold=None):
    """
    Gets an index of a word in the vocabulary and returns the indices of the word's nearest neighbours.
    :param src_word_idx: Index of the source words.
    :param embeddings: Embedding matrix, with shape (vocab_size, embedding_dim)
    :param dist_metric: The metric according to which distance from other words is being calculated.
    The default is euclidean distance, since this is the metric used by Alzantot et al. in their attack.
    :param ret_count: Number of nearest neighbours to return.
    :param threshold: If not None, only neighbours with distance smaller than the threshold are returned.
    :return: List of nearest neighbours indices and a list of the corresponding distances.
    """
    src_embedding = embeddings[src_word_idx]
    distances = cdist(np.array([src_embedding]), embeddings, dist_metric)[0]
    neighbours_indices = np.argsort(distances)[1:ret_count+1]
    neighbours_dist = distances[neighbours_indices]
    if dist_metric == "euclidean":
        neighbours_dist = np.square(neighbours_dist)
    if threshold is not None:
        mask = np.where(neighbours_dist < threshold)
        return neighbours_indices[mask], neighbours_dist[mask]
    else:
        return neighbours_indices, neighbours_dist