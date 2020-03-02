import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
import data_utils
import glove_utils
import models
import display_utils
from goog_lm import LM
import lm_data_utils
import lm_utils
from attacks import GeneticAtack

SAMPLE_SIZE = 5000
TEST_SIZE = 200
MAX_LEN = 250
VOCAB_SIZE = 50000
SUCCESS_THRESHOLD = 0.25
LSTM_SIZE = 128


def get_attack_indices(dataset):
    """
    Sample indices from the test set, on which the attack will be applied.
    :param dataset: The dataset for the sentiment analysis task.
    :return: The sampled indices to attack.
    """
    test_indices = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
    test_len = []
    for i in range(SAMPLE_SIZE):
        test_len.append(len(dataset.test_seqs2[test_indices[i]]))
    print('Shortest sentence in our test set is %d words' % np.min(test_len))
    return test_indices


def attack_model(model, ga_attack, test_x, test_y, indices):
    """
    Applies the attack of Alzantot et al.
    :param model: The sentiment analysis model to attack.
    :param ga_attack: Genetic attack model.
    :param test_x: The test sentences.
    :param test_y: The test labels.
    :param indices: Indices of the test sentences used by the attack.
    :return: Attack results.
    """
    test_list = []
    orig_list = []
    orig_label_list = []
    adv_list = []
    dist_list = []

    for i, sentence_idx in enumerate(indices):
        x_orig = test_x[sentence_idx]
        orig_label = test_y[sentence_idx]
        orig_preds = model.predict(sess, x_orig[np.newaxis, :])[0]
        if np.argmax(orig_preds) != orig_label:
            continue
        x_len = np.sum(np.sign(x_orig))
        if x_len >= 100:
            continue
        print('****** ', len(test_list) + 1, ' ********')
        test_list.append(sentence_idx)
        orig_list.append(x_orig)
        target_label = 1 if orig_label == 0 else 0
        orig_label_list.append(orig_label)
        x_adv = ga_attack.attack(x_orig, target_label)
        adv_list.append(x_adv)
        if x_adv is None:
            print('%d failed' % (i + 1))
            dist_list.append(100000)
        else:
            num_changes = np.sum(x_orig != x_adv)
            print('%d - %d changed.' % (i + 1, num_changes))
            dist_list.append(num_changes)
            # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
        print('--------------------------')
        if (len(test_list) >= TEST_SIZE):
            break
    return orig_list, dist_list, test_list, orig_label_list, adv_list


def compute_success_rate(orig_list, dist_list, test_list, orig_label_list, adv_list, model_name):
    """
    Computes Attack success rate.
    Parameters are the attack results, as returned by the attack_model function.
    :return: normalized_dist_list: For each test sentence used by the attack,
    the fraction of words that the attack replaced.
    successful_indices: Out of the test sentences used by the attack, on how many the attack succeeded.
    """
    orig_len = [np.sum(np.sign(x)) for x in orig_list]
    normalized_dist_list = [dist_list[i] / orig_len[i] for i in range(len(orig_list))]
    successful_attacks = [x < SUCCESS_THRESHOLD for x in normalized_dist_list]
    print('Attack success rate : {:.2f}%'.format(np.mean(successful_attacks) * 100))
    print('Median percentange of modifications: {:.02f}% '.format(
        np.median([x for x in normalized_dist_list if x < 1]) * 100))
    print('Mean percentange of modifications: {:.02f}% '.format(
        np.mean([x for x in normalized_dist_list if x < 1]) * 100))

    # visual_idx = np.random.choice(len(orig_list))
    # display_utils.visualize_attack(sess, model, dataset, orig_list[visual_idx], adv_list[visual_idx])

    test_list_np = np.array(test_list)
    successful_indices = test_list_np[successful_attacks]

    ## Save success
    with open('attack_results_{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump((test_list, orig_list, orig_label_list, adv_list, normalized_dist_list), f)

    np.save('success_indices_{}'.format(model_name), successful_indices)
    return normalized_dist_list, successful_indices


def load_model(model_path):
    """
    Load sentiment analysis model.
    :param model_path: Path to the model.
    :return: The model and the tensorflow session.
    """
    tf.reset_default_graph()
    if tf.get_default_session():
        sess.close()
    sess = tf.Session()
    batch_size = 1

    with tf.variable_scope('imdb', reuse=False):
        model = models.SentimentModel(batch_size=batch_size,
                                      lstm_size=LSTM_SIZE,
                                      max_len=MAX_LEN,
                                      embeddings_dim=300, vocab_size=VOCAB_SIZE + 1, is_train=False)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    return model, sess


def get_ga_attack(model, sess):
    """
    Get a genetic attack instance, initialized according to Alzantot et al.
    :param model: Sentiment analysis model to attack.
    :param sess: Tensoeflow session.
    :return: a genetic attack instance.
    """
    pop_size = 60
    n1 = 8

    with tf.variable_scope('imdb', reuse=True):
        batch_model = models.SentimentModel(batch_size=pop_size,
                                            lstm_size=LSTM_SIZE,
                                            max_len=MAX_LEN,
                                            embeddings_dim=300, vocab_size=VOCAB_SIZE+1, is_train=False)

    with tf.variable_scope('imdb', reuse=True):
        neighbour_model = models.SentimentModel(batch_size=n1,
                                                lstm_size=LSTM_SIZE,
                                                max_len=MAX_LEN,
                                                embeddings_dim=300, vocab_size=VOCAB_SIZE+1, is_train=False)
    ga_atttack = GeneticAtack(sess, model, batch_model, neighbour_model, dataset, embedding_matrix,
                              skip_list,
                              goog_lm, max_iters=30,
                              pop_size=pop_size,

                              n1=n1,
                              n2=4,
                              use_lm=True, use_suffix=False)
    return ga_atttack


if __name__ == '__main__':
    """
    Run the attack of Alzantot et al. 
    The attack targets sentiment analysis classifier. 
    """

    model_name = 'imdb_model_eps_8_sen_25'  # To use a different sentiment analysis model, change path
    print(model_name)
    model_path = './models/' + model_name

    np.random.seed(1001)
    tf.set_random_seed(1001)
    with open('aux_files/dataset_%d.pkl' % VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)

    embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' % (VOCAB_SIZE))).T
    skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' % VOCAB_SIZE)

    ### Preparing the dataset
    train_x = pad_sequences(dataset.train_seqs2, maxlen=MAX_LEN, padding='post')
    train_y = np.array(dataset.train_y)
    test_x = pad_sequences(dataset.test_seqs2, maxlen=MAX_LEN, padding='post')
    test_y = np.array(dataset.test_y)

    ### Loading the sentiment analysis model
    model, sess = load_model(model_path)

    ## Loading the Google Language model
    goog_lm = LM()

    ## Main Attack
    ga_attack = get_ga_attack(model, sess)


    # attack_indices = np.load('success_indices.npy')
    attack_indices = get_attack_indices(dataset)
    print("Used get attack indices")
    orig_list, dist_list, test_list, orig_label_list, adv_list = attack_model(model, ga_attack, test_x, test_y,
                                                                              attack_indices)
    normalized_dist_list, successful_indices = compute_success_rate(orig_list, dist_list, test_list, orig_label_list, adv_list, model_name)
    print(attack_indices)
    print(successful_indices)