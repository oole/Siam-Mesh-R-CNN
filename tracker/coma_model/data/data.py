import glob
import os

import numpy as np
from opendr.topology import get_vertices_per_edge
from psbody.mesh import Mesh
from sklearn.decomposition import PCA
from tensorpack.dataflow import DataFromList, MultiThreadMapData, BatchData, MapData
from tracker.config import config as cfg


class ChokepointDataflowProvider(object):
    def __init__(self, fit_pca=False, mesh_path=None):
        # self.mean = np.load(cfg.COMA.MESH_MEAN, allow_pickle=True)
        # self.std = np.load(cfg.COMA.MESH_STD, allow_pickle=True)
        # self.mean = np.load("../data/mesh_mean.npy", allow_pickle=True)
        # self.std = np.load("../data/mesh_std.npy", allow_pickle=True)

        self.train_mesh_paths = load_choke_meshes_paths(subset="train")
        self.val_mesh_paths = load_choke_meshes_paths(subset="val")
        self.test_mesh_paths = load_choke_meshes_paths(subset="test")

        # self.train_meshes = np.asarray(
        #     [self.normalize_mesh(np.load(mesh_path, allow_pickle=True)) for mesh_path in self.train_mesh_paths])
        # self.val_meshes = np.asarray(
        #     [self.normalize_mesh(np.load(mesh_path, allow_pickle=True)) for mesh_path in self.val_mesh_paths])
        # self.test_meshes = np.asarray(
        #     [self.normalize_mesh(np.load(mesh_path, allow_pickle=True)) for mesh_path in self.test_mesh_paths])

        self.train_meshes = np.asarray(
            [np.load(mesh_path, allow_pickle=True) for mesh_path in self.train_mesh_paths])
        self.val_meshes = np.asarray(
            [np.load(mesh_path, allow_pickle=True) for mesh_path in self.val_mesh_paths])
        self.test_meshes = np.asarray(
            [np.load(mesh_path, allow_pickle=True) for mesh_path in self.test_mesh_paths])

        self.mean = np.mean(self.train_meshes, axis=0)
        self.std = np.std(self.train_meshes, axis=0)

        self.train_meshes = self.normalize_mesh(self.train_meshes)
        self.val_meshes = self.normalize_mesh(self.val_meshes)
        self.test_meshes = self.normalize_mesh(self.test_meshes)

        if mesh_path is None:
            self.template_mesh = Mesh(filename=cfg.COMA.TEMPLATE_MESH)
        else:
            self.template_mesh = Mesh(filename=mesh_path)

        self.edge_vertices = get_vertices_per_edge(self.template_mesh.v, self.template_mesh.f)
        self.pca = PCA(n_components=8)#cfg.COMA.SIZE_LATENT)
        self.n_vertex = self.train_meshes.shape[1]
        if fit_pca:
            self.pca.fit(np.reshape(self.train_meshes, (self.train_meshes.shape[0], self.n_vertex * 3)))

    def normalize_mesh(self, mesh):
        mesh = mesh - self.mean
        mesh = mesh / self.std
        return mesh

    def get_fpn_feature_mesh_train_dataflow(self):
        ds = DataFromList(self.train_mesh_paths, shuffle=True)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            fpn_feature_path = mesh_path.replace("mesh_annotation", "bbox_features")
            fpn_feature = np.squeeze(np.squeeze(np.load(fpn_feature_path, allow_pickle=True), axis=0), axis=0)
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)

            return {'coma_fpn_feature_input': fpn_feature, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_fpn_feature_mesh_val_dataflow(self):
        ds = DataFromList(self.val_mesh_paths, shuffle=False)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            fpn_feature_path = mesh_path.replace("mesh_annotation", "bbox_features")
            fpn_feature = np.squeeze(np.squeeze(np.load(fpn_feature_path, allow_pickle=True), axis=0), axis=0)
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)

            return {'coma_fpn_feature_input': fpn_feature, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_fpn_feature_mesh_test_dataflow(self):
        ds = DataFromList(self.test_mesh_paths, shuffle=False)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            fpn_feature_path = mesh_path.replace("mesh_annotation", "bbox_features")
            fpn_feature = np.squeeze(np.squeeze(np.load(fpn_feature_path, allow_pickle=True), axis=0), axis=0)
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)

            return [fpn_feature, mesh]

        ds = MapData(ds, preprocess)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_fpn_feature_mesh_train_test_dataflow(self):
        ds = DataFromList(self.train_mesh_paths, shuffle=False)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            fpn_feature_path = mesh_path.replace("mesh_annotation", "bbox_features")
            fpn_feature = np.squeeze(np.squeeze(np.load(fpn_feature_path, allow_pickle=True), axis=0), axis=0)
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)

            return [fpn_feature, mesh]

        ds = MapData(ds, preprocess)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_fpn_feature_mesh_val_test_dataflow(self):
        ds = DataFromList(self.val_mesh_paths, shuffle=False)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            fpn_feature_path = mesh_path.replace("mesh_annotation", "bbox_features")
            fpn_feature = np.squeeze(np.squeeze(np.load(fpn_feature_path, allow_pickle=True), axis=0), axis=0)
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)

            return [fpn_feature, mesh]

        ds = MapData(ds, preprocess)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    ### For Autoencoder (aka Mesh2Feature)
    def get_mesh_mesh_train_dataflow(self):
        ds = DataFromList(self.train_meshes, shuffle=True)

        def preprocess(mesh):
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_mesh_mesh_val_dataflow(self):
        ds = DataFromList(self.val_meshes, shuffle=False)

        def preprocess(mesh):
            return [mesh, mesh]

        ds = MapData(ds, preprocess)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds


def load_choke_meshes_paths(subset=None):
    mesh_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "mesh_annotation", "G1", subset)
    paths_1 = sorted(glob.glob(mesh_path + "*/*/*/*/*"))
    meshes_1 = [path for path in paths_1 if ".npy" in path]
    paths_2 = sorted(glob.glob(mesh_path + "*/*/*/*"))
    meshes_2 = [path for path in paths_2 if ".npy" in path]
    mesh_paths = list(meshes_1 + meshes_2)
    sorted(mesh_paths)
    return mesh_paths
