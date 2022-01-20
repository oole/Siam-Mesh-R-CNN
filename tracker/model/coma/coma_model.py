import numpy as np
import scipy
import tensorflow._api.v2.compat.v1 as tf
from psbody.mesh import Mesh
from tensorpack.models import Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, FullyConnected
from tensorpack.tfutils import argscope

from coma.util import mesh_sampling, graph_util
from tracker.config import config as cfg
from tracker.model.coma import resnet_model
from tracker.model.util import spiral_util


class CoMa(object):
    def __init__(self, mesh_path=None, conv_mode="cheb"):
        if conv_mode not in ["cheb", "spiral"]:
            raise Exception("Unknown conv_mode '{}'".format(conv_mode))
        self.conv_mode = conv_mode
        if mesh_path is None:
            self.template_mesh = Mesh(filename=cfg.COMA.TEMPLATE_MESH)
        else:
            self.template_mesh = Mesh(filename=mesh_path)

        self.downsampling_factors = cfg.COMA.DOWNSAMPLING_FACTORS

        meshes, adjecency_m, downsampling_m, upsampling_m, sampling_faces, sampling_vertices = \
            mesh_sampling.get_transformation_matrices(self.template_mesh, self.downsampling_factors)

        self.adjecency_m = [x.astype('float32') for x in adjecency_m]  # convertType(adjecency_matrices)
        self.downsampling_m = [x.astype('float32') for x in downsampling_m]
        self.upsampling_m = [x.astype('float32') for x in upsampling_m]
        self.p = [x.shape[0] for x in adjecency_m]  # pooling size

        self.laplacians = [graph_util.laplacian(matrix, normalized=True) for matrix in self.adjecency_m]
        self.M_0 = self.laplacians[0].shape[0]

        # precompute spirals:
        self.spirals = spiral_util.process_spiral(sampling_faces, sampling_vertices, cfg.COMA.SPIRAL_LENGTH,
                                                  cfg.COMA.SPIRAL_DILATION)

        self.graph_input_features = cfg.COMA.GRAPH_INPUT_FEATURES  # 3
        self.graph_convolutional_filters = cfg.COMA.GRAPH_CONV_FILTERS  # [16, 16, 16, 32]
        self.poly_orders = cfg.COMA.GRAPH_POLY_ORDERS  # [6, 6, 6, 6]
        self.size_latent = cfg.COMA.SIZE_LATENT  # 8

        self.num_vertices = cfg.COMA.NUM_VERTICES  # 5023

        self.regularization = cfg.COMA.REGULARIZATION  # 5e-4
        self.dropout = cfg.COMA.DROPOUT  # 1
        self.learning_rate = cfg.COMA.LR  # 8e-3
        self.decay_rate = cfg.COMA.DECAY_RATE  # 0.99
        self.momentum = cfg.COMA.MOMENTUM  # 0.9
        self.batch_size = cfg.COMA.BATCH_SIZE
        # self.decay_steps = cfg.COMA.DECAY_STEPS  # num traine examples / batch size

        self.regularizers = []

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

    def loss(self, outputs, labels, edge_indices=None):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('coma_loss'):
            # L1
            with tf.name_scope('data_loss'):
                data_loss = tf.losses.absolute_difference(predictions=outputs, labels=labels,
                                                          reduction=tf.losses.Reduction.MEAN)

            if edge_indices is not None:
                with tf.name_scope("edge_loss"):
                    outputs_first_verts = tf.gather(outputs, edge_indices[:, 0], axis=1)
                    outputs_second_verts = tf.gather(outputs, edge_indices[:, 1], axis=1)
                    outputs_edge_lengths = tf.norm(outputs_first_verts - outputs_second_verts, axis=-1)

                    labels_first_verts = tf.gather(labels, edge_indices[:, 0], axis=1)
                    labels_second_verts = tf.gather(labels, edge_indices[:, 1], axis=1)
                    labels_edge_lengths = tf.norm(labels_first_verts - labels_second_verts, axis=-1)

                    edge_loss = tf.losses.absolute_difference(predictions=outputs_edge_lengths,
                                                              labels=labels_edge_lengths,
                                                              reduction=tf.losses.Reduction.MEAN)

            # with tf.name_scope('edge_length'):
            #     # transform to edges

            with tf.name_scope('regularization'):
                self.regularization *= tf.add_n(self.regularizers, name="regularization")

            if edge_indices is not None:
                loss = tf.add(data_loss, edge_loss, name="total_loss")
                tf.summary.scalar('edge_loss', data_loss)
                loss = tf.add(loss, self.regularization, name="total_loss")

            else:
                loss = tf.add(data_loss, self.regularization, name="total_loss")

            tf.summary.scalar('data_loss', data_loss)
            tf.summary.scalar('regularization', self.regularization)
            tf.summary.scalar('total', loss)
            with tf.name_scope('coma_averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)

                if edge_indices is not None:
                    op_averages = averages.apply([data_loss, self.regularization, loss, edge_loss])
                    tf.summary.scalar('avg/edge_loss', averages.average(edge_loss))
                else:
                    op_averages = averages.apply([data_loss, self.regularization, loss])
                tf.summary.scalar('avg/data_loss', averages.average(data_loss))
                tf.summary.scalar('avg/regularization', averages.average(self.regularization))
                tf.summary.scalar('avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def chebyshev5(self, x, L, Fout, K, name=None):
        _, M, Fin = x.get_shape()
        M, Fin = int(M), int(Fin)
        # N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph_util.rescale_laplacian(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        # print("Shape before transpose: {}".format(x.shape))
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N FIXME to:
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
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

        # x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N FIXME TO:
        x = tf.reshape(x, [K, M, Fin, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        # x = tf.reshape(x, [N * M, Fin * K])  # N*M x Fin*K FIXME TO:
        x = tf.reshape(x, [-1, Fin * K])
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin * K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        # return tf.reshape(x, [N, M, Fout])  # N x M x Fout FIXME TO:
        if name is None:
            return tf.reshape(x, [-1, M, Fout])
        else:
            return tf.reshape(x, [-1, M, Fout], name=name)

    def spiral_conv(self, x, spiral_indices, output_features, name=None):
        _, M, Fin = x.get_shape()
        M, Fin = int(M), int(Fin)
        spiral_length = spiral_indices.shape[1]

        x = tf.gather(x, spiral_indices.reshape(-1), axis=1)
        x = tf.reshape(x, [-1, M, int(x.get_shape()[1] * Fin) // M])

        w = self._weight_variable([Fin * spiral_length, output_features], regularization=False)
        x = tf.matmul(x, w)

        x = tf.reshape(x, [-1, M, output_features], name=name)
        return x

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        _, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        _, M, Fin = x.get_shape()
        M, Fin = int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # x = tf.reshape(x, [M, Fin * N])  # M x Fin*N FIXME TO:
        x = tf.reshape(x, [M, -1])
        x = tf.sparse_tensor_dense_matmul(L, x)  # Mp x Fin*N
        # x = tf.reshape(x, [Mp, Fin, N])  # Mp x Fin x N FIXME TO:
        x = tf.reshape(x, [Mp, Fin, -1])  # Mp x Fin x N
        x = tf.transpose(x, perm=[2, 0, 1])  # N x Mp x Fin

        return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        _, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def encoder(self, x, reuse=False, num_latent=None, conv_mode=None):
        with tf.variable_scope('coma_encoder', reuse=reuse):
            _, Min, Fin = x.get_shape()
            for i in range(len(self.graph_convolutional_filters)):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    with tf.name_scope('filter'):
                        if (conv_mode is None and self.conv_mode == "cheb") or conv_mode == "cheb":
                            x = self.chebyshev5(x, self.laplacians[i], self.graph_convolutional_filters[i],
                                                self.poly_orders[i])
                            # FIXME for debugging
                            # print(self.laplacians[i], self.graph_convolutional_filters[i], self.poly_orders[i])
                        else:
                            x = self.spiral_conv(x, self.spirals[i], self.graph_convolutional_filters[i])
                    with tf.name_scope('bias_relu'):
                        x = self.b1relu(x)
                    with tf.name_scope('pooling'):
                        x = self.poolwT(x, self.downsampling_m[i])

            # Fully connected hidden layers.
            # N, M, F = x.get_shape()
            # x = tf.reshape(x, [int(N), int(self.p[-1] * self.graph_convolutional_filters[-1])])  # N x MF FIXME TO:
            x = tf.reshape(x, [-1, int(self.p[-1] * self.graph_convolutional_filters[-1])])
            if self.size_latent:
                with tf.variable_scope('fc'):
                    if num_latent == None:
                        x = self.fc(x, int(self.size_latent))  # N x M0
                    else:
                        x = self.fc(x, int(num_latent))
        return x

    def decoder(self, x, reuse=False, conv_mode=None):
        with tf.variable_scope('coma_decoder', reuse=reuse):
            # N = x.get_shape()[0] # Removed dynamic batch size
            # M, F, Fin = self.D[-1].shape[0], self.F[-1], self.F_0
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1] * self.graph_convolutional_filters[-1]))  # N x MF

            # x = tf.reshape(x, [int(N), int(self.p[-1]), int(self.graph_convolutional_filters[-1])])  # N x M x F FIXME TO:
            x = tf.reshape(x, [-1, int(self.p[-1]), int(self.graph_convolutional_filters[-1])])  # N x M x F
            for i in range(len(self.graph_convolutional_filters)):
                with tf.variable_scope('upconv{}'.format(i + 1)):
                    with tf.name_scope('unpooling'):
                        x = self.poolwT(x, self.upsampling_m[-i - 1])
                    with tf.name_scope('filter'):
                        if (conv_mode is None and self.conv_mode == "cheb") or conv_mode == "cheb":
                            x = self.chebyshev5(x, self.laplacians[len(self.graph_convolutional_filters) - i - 1],
                                                self.graph_convolutional_filters[-i - 1], self.poly_orders[-i - 1])
                            # FIXME for debugging
                            # print(self.laplacians[-(i + 1)], self.graph_convolutional_filters[-(i + 1)],
                            #       self.poly_orders[-(i + 1)])
                        else:
                            x = self.spiral_conv(x, self.spirals[-i - 1], self.graph_convolutional_filters[-i - 1])
                    with tf.name_scope('bias_relu'):
                        x = self.b1relu(x)

            with tf.name_scope('outputs'):
                if (conv_mode is None and self.conv_mode == "cheb") or conv_mode == "cheb":
                    x = self.chebyshev5(x, self.laplacians[0], int(self.graph_input_features), self.poly_orders[0],
                                        name="reconstructed_mesh")
                else:
                    x = self.spiral_conv(x, self.spirals[0], self.graph_input_features, name="reconstructed_mesh", )
        return x

    def get_resnet_output(self, input_image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=cfg.COMA.DATA_FORMAT):
            return resnet_model.resnet_backbone(input_image,
                                                cfg.COMA.NUM_RESNET_BLOCKS,
                                                resnet_model.resnet_group,
                                                resnet_model.resnet_bottleneck,
                                                cfg.COMA.SIZE_LATENT)

    def img_to_mesh(self, input_image, reuse=False):
        with tf.variable_scope("coma_resnet", reuse=reuse):
            resnet_output = self.get_resnet_output(input_image)
        reconstructed_mesh = self.decoder(resnet_output, reuse=reuse)
        return reconstructed_mesh

    def img2lat(self, input_image, reuse=False):
        with tf.variable_scope("coma_resnet", reuse=reuse):
            resnet_output = self.get_resnet_output(input_image)
        return resnet_output

    def fpn_to_mesh(self, fpn_features, reuse=False, conv_mode=None):
        flatten = FullyConnected('flatten', fpn_features, 1024,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        fc = FullyConnected('fc', flatten, 32,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        return self.decoder(fc, reuse=reuse, conv_mode=conv_mode)
