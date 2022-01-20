import tensorflow._api.v2.compat.v1 as tf
from tensorpack import get_global_step_var
# from tracker.model.track_model import ResNetFPNTrackModel
from tensorpack.graph_builder import ModelDesc
from tensorpack.models import Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm
from tensorpack.tfutils import argscope
from tensorpack.tfutils.summary import add_moving_summary

from tracker.config import config as cfg
from tracker.model.coma import coma_model, resnet_model
from tracker.model.siam import base_model


class Coma(ModelDesc):
    def __init__(self, edge_vertices=None, is_eval=False, conv_mode="cheb", mesh_path=None):
        self.edge_vertices = edge_vertices
        self.is_eval = is_eval
        self.coma = coma_model.CoMa(conv_mode=conv_mode, mesh_path=mesh_path)

    def optimizer(self):
        lr = tf.train.exponential_decay(learning_rate=cfg.COMA.LR, global_step=get_global_step_var(),
                                        decay_steps=cfg.COMA.DECAY_STEPS, decay_rate=cfg.COMA.DECAY_RATE)
        tf.summary.scalar('learning_rate-summary', lr)
        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        opt = tf.train.MomentumOptimizer(lr, cfg.COMA.MOMENTUM)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        return ['coma_input_mesh'], ['coma_decoder/coma_outputs/reconstructed_mesh']

    def inputs(self):
        ret = []
        ret.append(tf.placeholder(tf.float32, (None, 5023, 3), "coma_input_mesh"))
        ret.append(tf.placeholder(tf.float32, (None, 5023, 3), "coma_target_mesh"))
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        input_mesh = inputs['coma_input_mesh']
        target_mesh = inputs['coma_target_mesh']

        with tf.variable_scope("mesh2feature"):
            latent = self.coma.encoder(input_mesh, reuse=False, num_latent=cfg.COMA.TRACK.ENCODER.FEATURE_SIZE)
            reconstructed_mesh = self.coma.decoder(latent, reuse=False)

        if self.training:
            cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)
            add_moving_summary(cost, moving_average)
            return cost
        elif self.is_eval:
            cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)


class ImageToMeshComa(ModelDesc):  # ResNetFPNTrackModel):

    def optimizer(self):
        lr = tf.train.exponential_decay(learning_rate=cfg.COMA.LR, global_step=get_global_step_var(),
                                        decay_steps=cfg.COMA.DECAY_STEPS, decay_rate=cfg.COMA.DECAY_RATE)
        tf.summary.scalar('learning_rate-summary', lr)
        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        opt = tf.train.MomentumOptimizer(lr, cfg.COMA.MOMENTUM)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        return ['coma_input_image'], ['coma_decoder/coma_outputs/reconstructed_mesh']

    def inputs(self):
        ret = []
        ret.append(tf.placeholder(tf.float32, (None, 224, 224, 3), "coma_input_image"))
        ret.append(tf.placeholder(tf.float32, (None, 5023, 3), "coma_target_mesh"))
        return ret

    def get_resnet_output(self, input_image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=cfg.COMA.DATA_FORMAT):
            return resnet_model.resnet_backbone(input_image,
                                                cfg.COMA.NUM_RESNET_BLOCKS,
                                                resnet_model.resnet_group,
                                                resnet_model.resnet_bottleneck,
                                                cfg.COMA.SIZE_LATENT)

    # NCHW vs NHWC, N batch, H height, W width, C channels
    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        input_image = inputs['coma_input_image']  # INPUT is NHWC
        preprocessed_image = base_model.image_preprocess(input_image, bgr=True)
        target_mesh = inputs['coma_target_mesh']

        coma = coma_model.CoMa()

        with tf.variable_scope("coma_resnet"):
            resnet_output = self.get_resnet_output(input_image)
        # latent = coma.encoder(input_mesh, reuse=False)
        reconstructed_mesh = coma.decoder(resnet_output, reuse=False)

        if self.training:
            cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)
            add_moving_summary(cost, moving_average)
            return cost
        elif self.is_eval:
            if self.edge_vertices is None:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh)
            else:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)
