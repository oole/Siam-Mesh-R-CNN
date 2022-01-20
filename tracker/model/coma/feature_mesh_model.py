import tensorflow._api.v2.compat.v1 as tf
from tensorpack import get_global_step_var
# from tracker.model.track_model import ResNetFPNTrackModel
from tensorpack.graph_builder import ModelDesc
from tensorpack.tfutils.summary import add_moving_summary

from tracker.config import config as cfg
from tracker.model.coma import coma_model


class FeatureToMeshComa(ModelDesc):
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
        return ['coma_fpn_feature_input'], ['coma_decoder/outputs/reconstructed_mesh']

    def inputs(self):
        ret = []
        ret.append(tf.placeholder(tf.float32, (None, 256, 7, 7), "coma_fpn_feature_input"))
        ret.append(tf.placeholder(tf.float32, (None, 5023, 3), "coma_target_mesh"))
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))

        input_features = inputs['coma_fpn_feature_input']
        target_mesh = inputs['coma_target_mesh']

        with tf.variable_scope("fpn2mesh"):
            reconstructed_mesh = self.coma.fpn_to_mesh(input_features)

        if self.training:
            if self.edge_vertices is None:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh)
                add_moving_summary(cost, moving_average)
                return cost
            else:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)
                add_moving_summary(cost, moving_average)
                return cost
        elif self.is_eval:
            if self.edge_vertices is None:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh)
            else:
                cost, moving_average = self.coma.loss(reconstructed_mesh, target_mesh, edge_indices=self.edge_vertices)
