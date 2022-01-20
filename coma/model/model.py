import collections
import os
import shutil
import time

import numpy as np
import scipy.sparse
import tensorflow._api.v2.compat.v1 as tf

import coma.util.graph_util as graph_util


class BaseModel(object):
    def __init__(self):
        self.regularizers = []
        # to be set by subclasses
        self.batch_size = None
        self.num_epochs = None
        self.dropout = None
        self.eval_frequency = None
        self.nz = None
        self.regularization = None
        self.learning_rate = None
        self.decay_steps = None
        self.decay_rate = None
        self.momentum = None
        self.which_loss = None
        self.dir_name = None

    # High-level interface which runs the constructed computational graph.

    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = [0] * size
        sess = self._get_session(sess)
        # sess = tf.Session(graph=self.graph)
        # sess.run(self.op_init)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, labels.shape[1], labels.shape[2]))
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]

        predictions = np.array(predictions)
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.perf_counter(), time.time()
        # print(labels.shape, data.shape)
        predictions, loss = self.predict(data, labels, sess)
        # print(predictions)
        # ncorrects = sum(predictions == labels)
        # accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
        # f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
        # string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
        #        accuracy, ncorrects, len(labels), f1, loss)
        string = 'loss: {:.2e}'.format(loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.clock() - t_process, time.time() - t_wall)
        return string, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.perf_counter(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), '')
        sess.run(self.op_init)

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx], train_labels[idx]
            if step == 1:
                print(batch_data.shape, batch_labels.shape)
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                # print(val_data.shape, val_labels.shape)
                string, loss = self.evaluate(val_data, val_labels, sess)
                # accuracies.append(accuracy)
                losses.append(loss)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.perf_counter() - t_process, time.time() - t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                # summary.value.add(tag='validation/accuracy', simple_value=accuracy)
                # summary.value.add(tag='validation/f1', simple_value=f1)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        # print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return losses, t_step

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.

    def build_graph(self, M_0, F_0):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('coma_inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, F_0), 'coma_data')
                self.ph_labels = tf.placeholder(tf.float32, (self.batch_size, M_0, F_0), 'coma_labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'coma_dropout')
                self.ph_z = tf.placeholder(tf.float32, (self.batch_size, self.nz[0]), 'coma_z')

            # Model.
            op_outputs = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_outputs, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = op_outputs
            self.op_encoder = self._encode(self.ph_data, reuse=True)
            self.op_decoder = self._decode(self.ph_z, reuse=True)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=200)

        #self.graph.finalize()

    def get_op_prediction(self):
        return self.op_prediction

    def get_ph_data(self):
        return self.ph_data

    def get_ph_dropout(self):
        return self.ph_dropout

    def get_graph(self):
        return self.graph

    def get_op_init(self):
        return self.op_init

    def encode(self, data):
        size = data.shape[0]
        z = [0] * size
        sess = self._get_session(sess=None)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

            batch_pred = sess.run(self.op_encoder, feed_dict)

            z[begin:end] = batch_pred[:end - begin]

        z = np.array(z)
        return z

    def decode(self, data):
        size = data.shape[0]
        x_rec = [0] * size
        sess = self._get_session(sess=None)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end, :]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end - begin] = tmp_data
            feed_dict = {self.ph_z: batch_data, self.ph_dropout: 1}

            batch_pred = sess.run(self.op_decoder, feed_dict)

            x_rec[begin:end] = batch_pred[:end - begin]

        x_rec = np.array(x_rec)
        return x_rec

    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, dropout)
        return logits

    # def probabilities(self, logits):
    #    """Return the probability of a sample to belong to each class."""
    #    with tf.name_scope('probabilities'):
    #        probabilities = tf.nn.softmax(logits)
    #        return probabilities

    # def prediction(self, logits):
    #    """Return the predicted classes."""
    #    with tf.name_scope('prediction'):
    #        prediction = tf.argmax(logits, axis=1)
    #        return prediction

    def loss(self, outputs, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('data_loss'):
                if self.which_loss == "l1":
                    data_loss = tf.losses.absolute_difference(predictions=outputs, labels=labels,
                                                              weights=self.loss_weights,
                                                              reduction=tf.losses.Reduction.MEAN)
                else:
                    data_loss = tf.losses.mean_squared_error(predictions=outputs, labels=labels,
                                                             weights=self.loss_weights,
                                                             reduction=tf.losses.Reduction.MEAN)
                # cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = data_loss + regularization

            # print(loss, l2_loss, regularization)
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/data_loss', data_loss)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([data_loss, regularization, loss])
                tf.summary.scalar('loss/avg/data_loss', averages.average(data_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='coma_global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='coma_control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        return os.path.join(self.model_path, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            # filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            # self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class CoMa(BaseModel):

    def __init__(self, L, D, U, F, K, p, nz, nv, which_loss, F_0=1, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                 unpool='poolwT', num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                 regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                 dir_name='', model_path="../computed/coma-model/"):
        super(CoMa, self).__init__()

        # Verify the consistency w.r.t. the number of layers.
        # assert len(L) >= len(F) == len(K) == len(p)
        # assert np.all(np.array(p) >= 1)
        # p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        # assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        # assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        # j = 0
        # self.L = []
        # for pp in p:
        #    self.L.append(L[j])
        #    j += int(np.log2(pp)) if pp > 1 else 0
        # L = self.L

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(nz)
        print('NN architecture')
        '''
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        '''
        # Store attributes and bind operations.
        self.L, self.D, self.U, self.F, self.K, self.p, self.nz, self.F_0 = L, D, U, F, K, p, nz, F_0
        self.which_loss = which_loss
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.model_path = model_path
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.unpool = getattr(self, unpool)

        # self.loss_weights = weight_tensor
        # Build the computational graph.
        self.build_graph(M_0, F_0)

    def loss(self, outputs, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('coma_loss'):
            with tf.name_scope('coma_data_loss'):
                if self.which_loss == "l1":
                    data_loss = tf.losses.absolute_difference(predictions=outputs, labels=labels,
                                                              reduction=tf.losses.Reduction.MEAN)
                else:
                    data_loss = tf.losses.mean_squared_error(predictions=outputs, labels=labels,
                                                             reduction=tf.losses.Reduction.MEAN)
                # cross_entropy = tf.reduce_mean(cross_entropy)
            with tf.name_scope('coma_regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = data_loss + regularization

            # print(loss, l2_loss, regularization)
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/data_loss', data_loss)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([data_loss, regularization, loss])
                tf.summary.scalar('loss/avg/data_loss', averages.average(data_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph_util.rescale_laplacian(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        print("Shape before transpose: {}".format(x.shape))
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin * N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _encode(self, x, reuse=False):
        with tf.variable_scope('coma_encoder', reuse=reuse):
            N, Min, Fin = x.get_shape()
            for i in range(len(self.F)):
                with tf.variable_scope('coma_conv{}'.format(i + 1)):
                    with tf.name_scope('coma_filter'):
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                        print(self.L[i], self.F[i], self.K[i])
                    with tf.name_scope('coma_bias_relu'):
                        x = self.brelu(x)
                    with tf.name_scope('coma_pooling'):
                        x = self.pool(x, self.D[i])

            # Fully connected hidden layers.
            # N, M, F = x.get_shape()
            x = tf.reshape(x, [int(N), int(self.p[-1] * self.F[-1])])  # N x MF
            if self.nz:
                with tf.variable_scope('coma_fc'):
                    x = self.fc(x, int(self.nz[0]))  # N x M0
        return x

    def _decode(self, x, reuse=False):
        with tf.variable_scope('coma_decoder', reuse=reuse):
            N = x.get_shape()[0]
            # M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            with tf.variable_scope('coma_fc2'):
                x = self.fc(x, int(self.p[-1] * self.F[-1]))  # N x MF

            x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.F[-1])])  # N x M x F

            for i in range(len(self.F)):
                with tf.variable_scope('coma_upconv{}'.format(i + 1)):
                    with tf.name_scope('coma_unpooling'):
                        x = self.unpool(x, self.U[-i - 1])
                    with tf.name_scope('coma_filter'):
                        x = self.filter(x, self.L[len(self.F) - i - 1], self.F[-i - 1], self.K[-i - 1])
                        print(self.L[-(i + 1)], self.F[-(i + 1)], self.K[-(i + 1)])
                    with tf.name_scope('coma_bias_relu'):
                        x = self.brelu(x)

            with tf.name_scope('coma_outputs'):
                x = self.filter(x, self.L[0], int(self.F_0), self.K[0])

        return x

    def _inference(self, x, dropout):
        z = self._encode(x)
        x = self._decode(z)

        # Graph convolutional layers.
        # x = tf.expand_dims(x, 2)  # N x M x F=1

        # Logits linear layer, i.e. softmax without normalization.
        # with tf.variable_scope('logits'):
        #    x = self.fc(x, self.M[-1], relu=False)
        return x

