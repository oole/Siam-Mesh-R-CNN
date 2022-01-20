import tensorflow._api.v2.compat.v1 as tf
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.varreplace import freeze_variables

from tracker.config import config as cfg
from tracker.model.coma import coma_model


class MeshModel(object):
    def __init__(self, mesh_path=None, fpn_conv_mode="spiral", coma_conv_mode="spiral"):
        self.coma = coma_model.CoMa(mesh_path=mesh_path, conv_mode=fpn_conv_mode)
        self.fpn_conv_mode = fpn_conv_mode
        self.coma_conv_mode = coma_conv_mode
        self.img2mesh_scope = None
        self.mesh2feature_scope = None
        self.fpn2mesh_scope = None
        with freeze_variables(stop_gradient=False, skip_collection=True):
            with tf.variable_scope("img2mesh"):
                self.img2mesh_scope = tf.get_variable_scope()
            with tf.variable_scope("mesh2feature"):
                self.mesh2feature_scope = tf.get_variable_scope()
            with tf.variable_scope("fpn2mesh"):
                self.fpn2mesh_scope = tf.get_variable_scope()

        self.img2mesh_init = False
        self.mesh2feature_init = False
        self.fpn2mesh_init = False

    @under_name_scope()
    def img2features(self, image_crops):
        with freeze_variables(stop_gradient=False, skip_collection=True):
            if self.img2mesh_scope is None:
                with tf.variable_scope("img2mesh"):
                    self.img2mesh_scope = tf.get_variable_scope()
                    mesh = self.coma.img_to_mesh(image_crops, reuse=False)
            else:
                with tf.variable_scope(self.img2mesh_scope, reuse=self.img2mesh_init):
                    mesh = self.coma.img_to_mesh(image_crops, reuse=self.img2mesh_init)
                    self.img2mesh_init = True

            if self.mesh2feature_scope is None:
                with tf.variable_scope("mesh2feature"):
                    self.mesh2feature_scope = tf.get_variable_scope()
                    mesh_feature = self.coma.encoder(mesh, reuse=False, num_latent=cfg.COMA.TRACK.ENCODER.FEATURE_SIZE)
            else:
                with tf.variable_scope(self.mesh2feature_scope, reuse=self.mesh2feature_init):
                    mesh_feature = self.coma.encoder(mesh, reuse=self.mesh2feature_init,
                                                     num_latent=cfg.COMA.TRACK.ENCODER.FEATURE_SIZE)
                    self.mesh2feature_init = True
        return mesh_feature

    @under_name_scope()
    def img2mesh(self, image_crops):
        with freeze_variables(stop_gradient=False, skip_collection=True):
            if self.img2mesh_scope is None:
                with tf.variable_scope("img2mesh"):
                    self.img2mesh_scope = tf.get_variable_scope()
                    mesh = self.coma.img_to_mesh(image_crops, reuse=False)
            else:
                with tf.variable_scope(self.img2mesh_scope, reuse=True):
                    mesh = self.coma.img_to_mesh(image_crops, reuse=True)
        return mesh

    @under_name_scope()
    def img2lat(self, image_crops):
        with freeze_variables(stop_gradient=False, skip_collection=True):
            if self.img2mesh_scope is None:
                with tf.variable_scope("img2mesh"):
                    self.img2mesh_scope = tf.get_variable_scope()
                    mesh = self.coma.img_to_mesh(image_crops, reuse=False)
            else:
                with tf.variable_scope(self.img2mesh_scope, reuse=True):
                    mesh = self.coma.img_to_mesh(image_crops, reuse=True)
        return mesh

    def fpn2features(self, fpn_features):
        with freeze_variables(stop_gradient=False, skip_collection=True):
            with tf.variable_scope(self.fpn2mesh_scope, reuse=self.fpn2mesh_init):
                mesh = self.coma.fpn_to_mesh(fpn_features, reuse=self.fpn2mesh_init, conv_mode=self.fpn_conv_mode)
                self.fpn2mesh_init = True

            with tf.variable_scope(self.mesh2feature_scope, reuse=self.mesh2feature_init):
                mesh_feature = self.coma.encoder(mesh, reuse=self.mesh2feature_init,
                                                 num_latent=cfg.COMA.TRACK.ENCODER.FEATURE_SIZE,
                                                 conv_mode=self.coma_conv_mode)
                self.mesh2feature_init = True
        return mesh_feature
