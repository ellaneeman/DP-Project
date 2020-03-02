from models import SentimentModel
import tensorflow as tf
import numpy as np
import math

class DPSentimentModel(SentimentModel):
    def __init__(self, dp_epsilon, dp_delta, attack_bound, n_draws=20, batch_size = 64, vocab_size=10000, max_len=200, lstm_size=64,
                 embeddings_dim=50, keep_probs=0.9, is_train=True):
        """
        A sentiment analysis classifier with an addition of a noise layer after the embedding layer.
        :param dp_epsilon: Epsilon parameter for DP bounds
        :param dp_delta: Delta parameter for DP bounds
        :param attack_bound: The fraction of words in a sentence that the attack can replace.
        :param n_draws: At prediction time, how many prediction are used to get the mean prediction value.

        *** The rest of the parameters are exactly as the ones in the classifier trained by Alzantot et al. ***
        """
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.attack_bound = attack_bound
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.lstm_size = lstm_size
        self.keep_probs = keep_probs
        self.embeddings_dim = embeddings_dim
        self.is_train = is_train
        self.n_draws = n_draws
        self.build_private_model()


    def _noise_layer(self, x, sensitivity_norm):
        """
        Pixeldp noise layer.
        :param x: The input for which noise is added.
        :param sensitivity_norm: String - can be l1 or l2.
        If l1, we use laplace noise. If l2, we use gaussian noise.
        :return: x + sampled noise.
        """
        input_shape = tf.shape(x)
        # experimental value
        word_sensitivity = tf.ones(input_shape, dtype=tf.float32) * 0.25
        sentence_sensitivity = self.attack_bound * self.seq_len

        if sensitivity_norm == 'l1':
            # Use the Laplace mechanism
            dp_mult = sentence_sensitivity / self.dp_epsilon
            loc = tf.zeros(input_shape, dtype=tf.float32)
            scale = tf.ones(input_shape, dtype=tf.float32)
            noise = tf.distributions.Laplace(loc, scale).sample()

        if sensitivity_norm == 'l2':
            # Use the Gaussian mechanism
            dp_mult = sentence_sensitivity * math.sqrt(2 * math.log(1.25 / self.dp_delta)) / self.dp_epsilon
            noise = tf.random_normal(input_shape, mean=0, stddev=1)

        dp_mult = tf.reshape(dp_mult, [tf.shape(dp_mult)[0], 1, 1])
        noise_scale = dp_mult * word_sensitivity
        noise = noise_scale * noise
        return x + noise

    def build_private_model(self):
        """
        Build a sentiment analysis classifier with an added noise layer after the embeddings layer.
        """
        # shape = (batch_size, sentence_length, word_id)
        self.x_holder = tf.placeholder(tf.int32, shape=[None, self.max_len])
        self.y_holder = tf.placeholder(tf.int64, shape=[None])
        self.seq_len = tf.cast(tf.reduce_sum(tf.sign(self.x_holder), axis=1), tf.float32)
        with tf.device("/cpu:0"):
            # embeddings matrix
            self.embedding_w = tf.get_variable('embed_w', shape=[self.vocab_size, self.embeddings_dim],
                                               initializer=tf.random_uniform_initializer(), trainable=True)


            # embedded words
            self.e = tf.nn.embedding_lookup(self.embedding_w, self.x_holder)

            self.noised_e = self._noise_layer(self.e, "l2")

        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        if self.is_train:
            self.noised_e = tf.nn.dropout(self.noised_e, self.keep_probs)
        self.init_state = lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=lstm,
                                                     inputs=self.noised_e,
                                                     initial_state=self.init_state,
                                                     sequence_length=self.seq_len)
        relevant = tf.reduce_mean(rnn_outputs, axis=1)
        last_output = relevant
        if self.is_train:
            last_output = tf.nn.dropout(last_output, self.keep_probs)
        self.w = tf.get_variable("w", shape=[self.lstm_size, 2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.2))
        self.b = tf.get_variable("b", shape=[2], dtype=tf.float32)
        logits = tf.matmul(last_output, self.w) + self.b
        self.y = tf.nn.softmax(logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.y_holder, depth=2), logits=logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_holder, tf.argmax(self.y, 1)), tf.float32))

        if self.is_train:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = self.optimizer.minimize(self.cost)

    def predict(self, sess, test_x):
        predictions = []
        for i in range(self.n_draws):
            # Similar to PixelDP, during prediction we call the
            # classifier several time and return the average prediction.
            pred_y = sess.run(self.y, feed_dict={self.x_holder: test_x})
            predictions.append(pred_y)
        res = np.mean(predictions, axis=0)
        return res

    def train_for_epoch(self, sess, train_x, train_y):
        assert self.is_train, 'Not training model'
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for idx in range(batches_per_epoch):
            batch_idx = np.random.choice(train_x.shape[0], size=self.batch_size, replace=False)
            batch_xs = train_x[batch_idx,:]
            batch_ys = train_y[batch_idx]
            batch_loss, _, batch_accuracy = sess.run([self.cost, self.train_op, self.accuracy],
                                     feed_dict={self.x_holder: batch_xs,
                                               self.y_holder: batch_ys})

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        return epoch_loss / batches_per_epoch, epoch_accuracy / batches_per_epoch