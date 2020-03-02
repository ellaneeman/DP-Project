import numpy as np
from scipy.spatial.distance import cdist
import glove_utils

NUM_NEIGHBOURS = 100


def explore_neighbourhood(dataset, embedding_matrix, n, iterations):
    """
    Randomly explore some (number of iterations) words' environments by printing topn nearest neighbors.
    This function is part of the exploration process to determine how many words are inside a meaningful neighborhood
    of a word.
    :param dataset: a dataset object containing an inv_full_dict to map a word's index to the word itself.
    :param embedding_matrix: a matrix containing pre-trained word embeddings with shape (emb_dim, vocab_size).
    :param topn: number of neighbors to print
    :param iterations: number of words to sample
    """
    print("explore")
    for i in range(iterations):
        print("------------------------")
        non_zero_indices = np.where(embedding_matrix.any(axis=1))[0]
        word_idx = np.random.choice(non_zero_indices, size=1)[0]
        print(word_idx)
        print("original word:", dataset.inv_full_dict[word_idx])
        print("------------------------")

        epsilon = 8
        sensitivity = 0.25
        chosen_indices, topn = sample_word(embedding_matrix.T[word_idx], embedding_matrix.T, epsilon,
                                                  sensitivity, n=n, size=10)
        print("nearest neighbors")
        for x_idx, x in enumerate(topn):
            print(x_idx, dataset.inv_full_dict[x])

        print("------------------------")
        print("original word:", dataset.inv_full_dict[word_idx])
        print("words and ranks by em")
        for idx in chosen_indices:
            print(dataset.inv_full_dict[idx])


def calculate_approx_sensitivity(embedding_matrix, topn=NUM_NEIGHBOURS, num_samples=1000, utility='cosine'):
    """
    Randomly samples words in the vocabulary, then for each word x and its first neighbor y calculates max/mean
    difference in the utility function's values inside x's neighborhood (excluding y).
    Finally, returns the expected value (mean).
    :param embedding_matrix: a matrix containing pre-trained word embeddings with shape (vocab_size, emb_dim).
    :param topn: number of neighbors inside a "meaningful neighborhood" (from the exploratory process).
    :param num_samples: number of words to consider in the approximation.
    :param utility: word embeddings similarity function (cosine or euclidean).
    :return: mean sensitivity approximation, max sensitivity approximation
    """
    topn = topn + 1
    non_zero_indices = np.where(embedding_matrix.any(axis=1))[0]
    embedding_matrix = embedding_matrix[non_zero_indices]

    mean_list, max_list, min_list = [], [], []  # min is calculated for sanity
    for i in range(num_samples):
        sample = embedding_matrix[np.random.choice(embedding_matrix.shape[0], size=1)]

        # calculate distances from x
        distances_x = cdist(sample, embedding_matrix, utility)[0]
        # x's nearest neighbors
        neighbours_indices = np.argsort(distances_x)[1:topn + 1]
        # y is the first neighbor of x
        closest_neighbour_idx = neighbours_indices[0]
        # x's neighborhood without y
        neighbours_indices = neighbours_indices[1:]
        # x's neighborhood distances from x
        neighbours_dist = distances_x[neighbours_indices]
        # x's neighborhood's embeddings (vector representations)
        x_neighbourhood = embedding_matrix[neighbours_indices]
        # x's neighborhood distances from y
        distances_y = cdist([embedding_matrix[closest_neighbour_idx]], x_neighbourhood, utility)[0]
        # differences in the utility function's values inside x's neighborhood (excluding y)
        diff = np.abs(neighbours_dist - distances_y)

        mean_list.append(np.mean(diff))
        max_list.append(np.max(diff))
        min_list.append(np.min(diff))

    return np.mean(mean_list), np.mean(max_list)


def sample_word(original_word, embedding_matrix, epsilon, sensitivity, n=NUM_NEIGHBOURS, utility='cosine', size=1):
    """
    Gets an original word index and samples a replacement word's index using the exponential mechanism.
    :param original_word: an integer representing a word in the vocabulary.
    :param embedding_matrix: a matrix containing pre-trained word embeddings with shape (vocab_size, emb_dim).
    :param epsilon: a float, dp param for the exponential mechanism.
    :param sensitivity: a float, dp param for the exponential mechanism.
    :param n: number of top nearest neighbors to take to the topn list.
    :param utility: the utility function for the the exponential mechanism. In our case, the similarity between words.
    :param size: number of indices to sample.
    :return: a replacement word's index.
    """
    distances = cdist(np.array([original_word]), embedding_matrix, utility)[0]
    # assign max distance in terms of cosine similarity to none values so they would never be sampled.
    # none values occur when a word has a zero embedding vector, which happens for words that didn't
    # appear in the pre-trained embeddings.
    distances[np.argwhere(np.isnan(distances))] = 2
    similarity = 1 - distances
    scores = np.exp(similarity * epsilon / (2 * sensitivity))
    probabilities = scores / np.sum(scores)

    topn_probs = np.argsort(probabilities)[::-1][:n]  # For printing examples

    chosen_indices = np.random.choice(np.arange(embedding_matrix.shape[0]), size=size, p=probabilities)

    # return chosen_indices, topn_probs  # For printing examples. If running explore_neighbourhood, uncomment this line
    return chosen_indices[0]