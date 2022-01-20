import numpy as np
from psbody.mesh import Mesh, MeshViewer
import time
import random
from copy import deepcopy
import glob
from sklearn.decomposition import PCA
from opendr.topology import get_vertices_per_edge


class MeshData(object):
    def __init__(self, number_val: int, train_file: str, test_file: str, reference_mesh_file: str, num_pca_components=8,
                 fit_pca=False):
        """
        Constructor; create a new Mesh Dataset, consisting of train, validation and test set.

        :param number_val: The number of validation samples from the training dataset
        :param train_file: The file containing the training samples
        :param test_file: The file containing the test samples
        :param reference_mesh_file: The path to the reference mesh file
        """


        self.number_val = number_val
        self.train_file = train_file
        self.test_file = test_file

        self.vertices_train_non_norm = None
        self.vertices_val_non_norm = None
        self.vertices_test_non_norm = None

        self.vertices_train = None
        self.vertices_val = None
        self.vertices_test = None
        self.N = None
        self.n_vertex = None
        self.mean = None
        self.std = None

        self.load()
        self.reference_mesh = Mesh(filename=reference_mesh_file)

        self.edge_vertices = get_vertices_per_edge(self.reference_mesh.v, self.reference_mesh.f)

        self.pca = PCA(n_components=num_pca_components)
        self.fit_pca = fit_pca

        self.normalize()



    def load(self):
        """
        Load the mesh data from the train_file and test_file. The train_file will be split into training and validation
        by splitting the train_file, according to the numer of validation samples.
        """
        vertices_train = np.load(self.train_file)
        vertices_test = np.load(self.test_file)

        self.vertices_train_non_norm = vertices_train[:-self.number_val]
        self.vertices_val_non_norm = vertices_train[-self.number_val:]
        self.vertices_test_non_norm = vertices_test
        self.mean = np.mean(vertices_train, axis=0)
        self.std = np.std(vertices_train, axis=0)

        # split in to train val
        self.vertices_train = vertices_train[:-self.number_val]
        self.vertices_val = vertices_train[-self.number_val:]

        self.n_vertex = self.vertices_train.shape[1]

        self.vertices_test = vertices_test

    def normalize(self):
        """
        Normalizes the dataset by substracting the training set mean and dividing by the training
        set standard deviation.
        """
        # train
        self.vertices_train = self.vertices_train - self.mean
        self.vertices_train = self.vertices_train / self.std

        self.vertices_val = self.vertices_val - self.mean
        self.vertices_val = self.vertices_val / self.std

        self.vertices_test = self.vertices_test - self.mean
        self.vertices_test = self.vertices_test / self.std

        self.N = self.vertices_train.shape[0]

        if self.fit_pca:
            self.pca.fit(np.reshape(self.vertices_train, (self.N, self.n_vertex*3)))



    def show(self, ids):
        """
        Shows the provided meshes, given the ids provided.
        """
        if max(ids >= self.N):
            raise ValueError('id: out of bounds')

        mesh = Mesh(v=self.vertices_train[ids[0]], f=self.reference_mesh.f)
        time.sleep(0.5)
        viewer = mesh.show()
        for i in range(len(ids) - 1):
            viewer.dynamic_meshes = [Mesh(v=self.vertices_train[ids[i + 1]], f=self.reference_mesh.f)]
            time.sleep(0.5)
        return 0

    def sample(self, BATCH_SIZE=64):
        """
        Randomly samples the training vertices given the batch size
        """
        samples = np.zeros((BATCH_SIZE, self.vertices_train.shape[1] * self.vertices_train.shape[2]))
        for i in range(BATCH_SIZE):
            randint = random.randint(0, self.N - 1)
            samples[i] = ((deepcopy(self.vertices_train[randint]) - self.mean) / self.std).reshape(-1)

        return samples

    def save_meshes(self, filename, meshes):
        """
        Stores the given meshes after de-normalizing it.

        Filepath pattern: filename + '-' + str(i).zfill(3) + 'ply'
        """
        for i in range(meshes.shape[0]):
            vertices = meshes[i].reshape((self.n_vertex, 3)) * self.std + self.mean
            mesh = Mesh(v=vertices, f=self.reference_mesh.f)
            mesh.write_ply(filename + '-' + str(i).zfill(3) + '.ply')
        return 0

    def show_mesh(self, viewer, mesh_vecs, figsize):
        """
        Displays the given mesh_vectors in a mesh viewer, given the figsize.
        """
        for i in range(figsize[0]):
            for j in range(figsize[1]):
                mesh_vec = mesh_vecs[i * figsize[0] - 1 + j]
                mesh_mesh = self.vec2mesh(mesh_vec)
                viewer[i][j].set_dynamic_meshes([mesh_mesh])
        time.sleep(0.1)
        return 0

    def vec2mesh(self, vec):
        vec = vec.reshape((self.n_vertex, 3)) * self.std + self.mean
        return Mesh(v=vec, f=self.reference_mesh.f)


def meshPlay(folder, every=100, wait=0.05):
    """"
    Displays the meshes in the given folder in a MeshViewer
    """
    files = glob.glob(folder + '/*')
    files.sort()
    files = files[-1000:]
    view = MeshViewer()
    for i in range(0, len(files), every):
        mesh = Mesh(filename=files[i])
        view.dynamic_meshes = [mesh]
        time.sleep(wait)
