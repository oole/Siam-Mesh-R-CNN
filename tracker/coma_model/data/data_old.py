import glob
import os

import cv2
import numpy as np
from tensorpack.dataflow import (
    DataFromList, MultiThreadMapData, imgaug, BatchData)

from coma.data import meshdata
from tracker.config import config as cfg


class ChokepointDataflowProvider(object):
    def __init__(self):
        # self.mean = np.load(cfg.COMA.MESH_MEAN, allow_pickle=True)
        # self.std = np.load(cfg.COMA.MESH_STD, allow_pickle=True)
        self.mean = np.load("../data/mesh_mean.npy", allow_pickle=True)
        self.std = np.load("../data/mesh_std.npy", allow_pickle=True)

    def normalize_mesh(self, mesh):
        mesh = mesh - self.mean
        mesh = mesh / self.std
        return mesh

    def get_image_mesh_train_dataflow(self):
        mesh_paths = load_choke_meshes_paths(subset="train")
        ds = DataFromList(mesh_paths, shuffle=True)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            image_path = mesh_path.replace("mesh_annotation", "face_data").replace(".npy", ".jpg")
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # motion blur
            if cfg.DATA.MOTION_BLUR_AUGMENTATIONS:
                do_motion_blur = np.random.rand() < 0.25
                if do_motion_blur:
                    # generating the kernel
                    kernel_size = np.random.randint(5, 15)
                    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
                    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                    kernel_motion_blur = kernel_motion_blur / kernel_size
                    # applying the kernel
                    img = cv2.filter2D(img, -1, kernel_motion_blur)

            # grayscale
            if cfg.DATA.GRAYSCALE_AUGMENTATIONS:
                do_grayscale = np.random.rand() < 0.25
                if do_grayscale:
                    grayscale_aug = imgaug.Grayscale()
                    img = np.tile(grayscale_aug.augment(img), [1, 1, 3])

            return {'coma_input_image': img, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_image_mesh_val_dataflow(self):
        mesh_paths = load_choke_meshes_paths(subset="val")
        ds = DataFromList(mesh_paths, shuffle=True)

        def preprocess(mesh_path: str):
            # collect image and mesh and provide dict with input tensor names and input data
            image_path = mesh_path.replace("mesh_annotation", "face_data").replace(".npy", ".jpg")
            mesh = np.load(mesh_path, allow_pickle=True)
            mesh = self.normalize_mesh(mesh)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # motion blur
            if cfg.DATA.MOTION_BLUR_AUGMENTATIONS:
                do_motion_blur = np.random.rand() < 0.25
                if do_motion_blur:
                    # generating the kernel
                    kernel_size = np.random.randint(5, 15)
                    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
                    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                    kernel_motion_blur = kernel_motion_blur / kernel_size
                    # applying the kernel
                    img = cv2.filter2D(img, -1, kernel_motion_blur)

            # grayscale
            if cfg.DATA.GRAYSCALE_AUGMENTATIONS:
                do_grayscale = np.random.rand() < 0.25
                if do_grayscale:
                    grayscale_aug = imgaug.Grayscale()
                    img = np.tile(grayscale_aug.augment(img), [1, 1, 3])

            return {'coma_input_image': img, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_mesh_mesh_validation_dataflow(self):
        mesh_paths = self.load_meshes_paths(subset="val")
        meshes = np.asarray([np.load(path) for path in mesh_paths])
        meshes = self.normalize_mesh(meshes).astype('float32')
        ds = DataFromList(meshes, shuffle=True)

        def preprocess(mesh: str):
            # collect image and mesh and provide dict with input tensor names and input data
            # mesh = np.load(mesh_path, allow_pickle=True)
            # mesh = self.normalize_mesh(mesh)
            return mesh, mesh

        # ret = preprocess(mesh_paths[0])

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_mesh_mesh_train_dataflow(self):
        mesh_paths = self.load_meshes_paths(subset="train")
        meshes = np.asarray([np.load(path) for path in mesh_paths])
        meshes = self.normalize_mesh(meshes).astype('float32')
        ds = DataFromList(meshes, shuffle=True)

        def preprocess(mesh: str):
            # collect image and mesh and provide dict with input tensor names and input data
            # mesh = np.load(mesh_path, allow_pickle=True)
            # mesh = self.normalize_mesh(mesh)
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        # ret = preprocess(mesh_paths[0])

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds


def load_choke_meshes_paths(subset=None):
    mesh_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "mesh_annotation", "G1", subset)
    paths_1 = sorted(glob.glob(mesh_path + "*/*/*/*/*"))
    meshes_1 = [path for path in paths_1 if ".npy" in path]
    paths_2 = sorted(glob.glob(mesh_path + "*/*/*/*"))
    meshes_2 = [path for path in paths_2 if ".npy" in path]
    mesh_paths = list(meshes_1 + meshes_2)
    return mesh_paths


class ComaDataflowProvider(object):
    def __init__(self):
        mesh_data = meshdata.MeshData(number_val=100, train_file=cfg.COMA.COMA_DATA_PATH + '/train.npy',
                                      test_file=cfg.COMA.COMA_DATA_PATH + '/test.npy',
                                      reference_mesh_file=cfg.COMA.TEMPLATE_MESH)
        self.x_train = mesh_data.vertices_train.astype('float32')
        self.x_val = mesh_data.vertices_val.astype('float32')
        self.x_test = mesh_data.vertices_val.astype('float32')
        self.mean = mesh_data.mean
        self.std = mesh_data.std

    def get_coma_mesh_mesh_train_dataflow(self):
        ds = DataFromList(self.x_train, shuffle=True)

        def preprocess(mesh):
            mesh = mesh
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_coma_mesh_mesh_val_dataflow(self):
        ds = DataFromList(self.x_val, shuffle=True)

        def preprocess(mesh):
            mesh = mesh
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds


class AllDataflowProvider(object):
    def __init__(self):
        mesh_data = meshdata.MeshData(number_val=100, train_file=cfg.COMA.COMA_DATA_PATH + '/train.npy',
                                      test_file=cfg.COMA.COMA_DATA_PATH + '/test.npy',
                                      reference_mesh_file=cfg.COMA.TEMPLATE_MESH)

        self.x_train = mesh_data.vertices_train_non_norm.astype('float32')
        self.x_val = mesh_data.vertices_val_non_norm.astype('float32')
        self.x_test = mesh_data.vertices_test_non_norm.astype('float32')

        # self.mean = mesh_data.mean
        # self.std = mesh_data.std

        self.choke_train_meshes_paths = np.asarray([np.load(path) for path in load_choke_meshes_paths("train")])
        self.choke_val_meshes_paths = np.asarray([np.load(path) for path in load_choke_meshes_paths("val")])

        self.all_train_meshes = np.concatenate((self.x_train, self.choke_train_meshes_paths))
        self.all_val_meshes = np.concatenate((self.x_val, self.choke_val_meshes_paths))
        self.mean = np.load("../data/mesh_mean.npy", allow_pickle=True)
        self.std = np.load("../data/mesh_std.npy", allow_pickle=True)
        # No need to recomputed we stored this exact mesh_mean and mesh_std
        # self.mean, self.std = self.compute_mean_std(self.all_train_meshes)
        self.normalize_all()
        self.edge_vertices = mesh_data.edge_vertices

    def normalize_all(self):
        self.all_train_meshes = self.all_train_meshes - self.mean
        self.all_train_meshes = self.all_train_meshes / self.std

        self.all_val_meshes = self.all_val_meshes - self.mean
        self.all_val_meshes = self.all_val_meshes / self.std

    def compute_mean_std(self, train_meshes):
        mean = np.mean(train_meshes, axis=0)
        std = np.std(train_meshes, axis=0)
        return mean, std

    def get_all_mesh_mesh_train_dataflow(self):
        ds = DataFromList(self.all_train_meshes, shuffle=True)

        def preprocess(mesh):
            mesh = mesh
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds

    def get_all_mesh_mesh_val_dataflow(self):
        ds = DataFromList(self.all_val_meshes, shuffle=True)

        def preprocess(mesh):
            mesh = mesh
            return {'coma_input_mesh': mesh, "coma_target_mesh": mesh}

        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
        ds = BatchData(ds, cfg.COMA.BATCH_SIZE)
        return ds


if __name__ == "__main__":
    input_dataflow = get_mesh_image_train_dataflow()
