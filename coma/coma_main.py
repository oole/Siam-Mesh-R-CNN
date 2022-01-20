from __future__ import print_function

import argparse
import copy
import json
import os

import numpy as np
from psbody.mesh import Mesh

from coma.data import meshdata
from coma.util import mesh_sampling, graph_util, latent_magic
from coma.model import model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


parser = argparse.ArgumentParser(description='Tensorflow Trainer for Convolutional Mesh Autoencoders')
parser.add_argument('--name', default='test', help='facial_motion| lfw ')
parser.add_argument('--data', default='/mnt/storage/Msc/oole-coma-data/processed-data/sliced', help='facial_motion| lfw ')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs to train (default: 1)')
parser.add_argument('--eval_frequency', type=int, default=200, help='eval frequency')
parser.add_argument('--filter', default='chebyshev5', help='filter')
parser.add_argument('--nz', type=int, default=8, help='Size of latent variable')
parser.add_argument('--lr', type=float, default=8e-3, help='Learning Rate')
parser.add_argument('--workers', type=int, default=4, help='number of data loading threads')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--viz', type=int, default=0, help='visualize while test')
parser.add_argument('--loss', default='l1', help='l1 or l2')
parser.add_argument('--mesh1', default='m1', help='for mesh interpolation')
parser.add_argument('--mesh2', default='m1', help='for mesh interpolation')
parser.add_argument("--model-path", default="../computed/tf")

args = parser.parse_args()

np.random.seed(args.seed)
nz = args.nz
print("Loading data .. ")
template_mesh_path = '../data/template.obj'
base_data_folder = args.data

template_mesh = Mesh(filename=template_mesh_path)
mesh_data = meshdata.MeshData(number_val=100, train_file=base_data_folder + '/train.npy',
                              test_file=base_data_folder + '/test.npy',
                              reference_mesh_file=template_mesh_path)

downsampling_factors = [4, 4, 4, 4]  # Sampling factor of the mesh at each stage of sampling
print("Generating Transform Matrices ..")

# Generates adjecency matrices A, downsampling matrices D, and upsamling matrices U by sampling
# the mesh 4 times. Each time the mesh is sampled by a factor of 4


meshes, adjecency_matrices, downsampling_matrices, upsampling_matrices, sampling_faces, sampling_vertices = \
    mesh_sampling.get_transformation_matrices(template_mesh, downsampling_factors)

# convert dtypes
adjecency_matrices = [x.astype('float32') for x in adjecency_matrices]  # convertType(adjecency_matrices)
downsampling_matrices = [x.astype('float32') for x in downsampling_matrices]
upsampling_matrices = [x.astype('float32') for x in upsampling_matrices]
p = [x.shape[0] for x in adjecency_matrices]

# A: 5023x5023, 1256x1256, 314x314, 79x79, 20x20
# D: 1256x5023, 314x1256, 79x314, 20x79
# U: 5023x1256, 1256x314, 314x79, 79x20
# p: 5023, 1256, 314, 79, 20

print("Computing Graph Laplacians ..")
laplacians = [graph_util.laplacian(matrix, normalized=True) for matrix in adjecency_matrices]

x_train = mesh_data.vertices_train.astype('float32')
x_val = mesh_data.vertices_val.astype('float32')
x_test = mesh_data.vertices_test.astype('float32')

num_train_examples = x_train.shape[0]
params = dict()
params['dir_name'] = args.name
params['num_epochs'] = args.num_epochs
params['batch_size'] = args.batch_size
params['eval_frequency'] = args.eval_frequency
# Building blocks.
params['filter'] = args.filter
params['brelu'] = 'b1relu'
params['pool'] = 'poolwT'
params['unpool'] = 'poolwT'

# Architecture.
params['F_0'] = int(x_train.shape[2])  # Number of graph input features.
params['F'] = [16, 16, 16, 32]  # Number of graph convolutional filters.
params['K'] = [6, 6, 6, 6]  # Polynomial orders.
params['p'] = p  # [4, 4, 4, 4]    # Pooling sizes.
params['nz'] = [nz]  # Output dimensionality of fully connected layers.

# Optimization.
params['which_loss'] = args.loss
params['nv'] = mesh_data.n_vertex
params['regularization'] = 5e-4
params['dropout'] = 1
params['learning_rate'] = args.lr
params['decay_rate'] = 0.99
params['momentum'] = 0.9
params['decay_steps'] = num_train_examples / params['batch_size']
params['model_path'] = args.model_path

model = model.CoMa(L=laplacians, D=downsampling_matrices, U=upsampling_matrices, **params)

if args.mode in ['test']:
    if not os.path.exists('results'):
        os.makedirs('results')
    predictions, loss = model.predict(x_test, x_test)
    print("L1 Loss= ", loss)
    euclidean_loss = np.mean(np.sqrt(np.sum((mesh_data.std * (predictions - mesh_data.vertices_test)) ** 2, axis=2)))
    print("Euclidean loss= ", euclidean_loss)
    np.save('results/' + args.name + '_predictions', predictions)
elif args.mode in ['sample']:
    meshes = mesh_data.get_normalized_meshes(args.mesh1, args.mesh2)
    features = model.encode(meshes)
elif args.mode in ['latent']:
    latent_magic.play_with_latent_space(model, mesh_data)
else:
    if not os.path.exists(os.path.join(args.model_path, args.name)):
        os.makedirs(os.path.join(args.model_path, args.name))
    with open(os.path.join(args.model_path, "parameters", args.name + '-params.json'), 'w') as fp:
        saveparams = copy.deepcopy(params)
        saveparams['seed'] = args.seed
        json.dump(saveparams, fp)
    loss, t_step = model.fit(x_train, x_train, x_val, x_val)
