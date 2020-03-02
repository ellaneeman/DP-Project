"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import os
import data_utils

from keras.preprocessing.sequence import pad_sequences

import numpy as np
import tensorflow as tf
import pickle

import models
import exponential_mechanism

IMDB_PATH = 'aclImdb'
MAX_VOCAB_SIZE = 50000


def preprocess_data(sentence, attack_bound, embedding_matrix, epsilon, sensitivity):
    """
    Add noise to an input sentence using the exponential mechanism.
    :param sentence: The input sentence. The sentence is represented with the indices of the sentence words.
    The input sentence is zero-padded to be in a constant, predefined length.
    :param attack_bound: The fraction of words in a sentence that the attack can replace.
    :param embedding_matrix: Word embedding matrix.
    :param epsilon: Epsilon parameter for the Exponential Mechanism
    :param sensitivity: The sensitivity of the Exponential Mechanism utility function.
    In our case, the utility function is the similarity between a given word and other words in the vocabulary.
    :return:
    """
    sentence_len = np.sum(np.sign(sentence))  # Number of words in the original sentence (without the zero-padding)

    # For each word, we sample the from the binomial distribution with p set to be the attack bound.
    # If 1 was sampled, the word is replaced using the Exponential Mechanism.
    # For example, if the attack bound is 0.25, the Exponential Mechanism is applied to each word with probability 0.25.
    replace_indices = np.random.binomial(n=1, p=attack_bound, size=sentence_len)

    for i in range(sentence_len):
        if replace_indices[i]:
            # Sample a replacement word using the Exponential Mechanism
            chosen_idx = exponential_mechanism.sample_word(embedding_matrix[sentence[i]],
                                                           embedding_matrix, epsilon, sensitivity)
            sentence[i] = chosen_idx
    return sentence


if __name__ == '__main__':
    """
    Train a sentiment analysis classifier with noise addition.  
    The noise is appplied directly to the input layer
    """
    args = [[0.25, 8]]  # sensitivity, epsilon. To train the classifier with different values, change them here.

    with open(('aux_files/dataset_%d.pkl' % MAX_VOCAB_SIZE), 'rb') as f:
        dataset = pickle.load(f)
    embedding_matrix = np.load(('aux_files/embeddings_glove_%d.npy' % (MAX_VOCAB_SIZE)))
    max_len = 250

    train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')

    for sensitivity, epsilon in args:
        print('training a model with eps {} and sen {}'.format(epsilon, sensitivity))
        train_x = np.apply_along_axis(preprocess_data, axis=1, arr=np.array(train_x), attack_bound=0.25,
                                      embedding_matrix=embedding_matrix.T, epsilon=epsilon, sensitivity=sensitivity)

        train_y = np.array(dataset.train_y)
        test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
        test_y = np.array(dataset.test_y)
        sess = tf.Session()
        batch_size = 64
        lstm_size = 128
        num_epochs = 20
        with tf.variable_scope('imdb', reuse=False):
            model = models.SentimentModel(batch_size=batch_size,
                                          lstm_size=lstm_size,
                                          max_len=max_len,
                                          keep_probs=0.8,
                                          embeddings_dim=300, vocab_size=embedding_matrix.shape[1],
                                          is_train=True)
        with tf.variable_scope('imdb', reuse=True):
            test_model = models.SentimentModel(batch_size=batch_size,
                                               lstm_size=lstm_size,
                                               max_len=max_len, keep_probs=0.8,
                                               embeddings_dim=300, vocab_size=embedding_matrix.shape[1], is_train=False)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.embedding_w, embedding_matrix.T))
        print('Training..')
        for i in range(num_epochs):
            epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
            print(i, ' ', epoch_loss, ' ', epoch_accuracy)
            print('Train accuracy = ', test_model.evaluate_accuracy(sess, train_x, train_y))
            print('Test accuracy = ', test_model.evaluate_accuracy(sess, test_x, test_y))
        if not os.path.exists('models'):
            os.mkdir('models')
        saver = tf.train.Saver()
        saver.save(sess, 'models/imdb_model_eps_{}_sen_{}'.format(epsilon, int(sensitivity*100)))
    print('All done')