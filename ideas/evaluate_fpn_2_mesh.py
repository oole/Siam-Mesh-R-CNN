import argparse
import os

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tracker.coma_model.data import data
from tracker.model.coma import feature_mesh_model
from psbody.mesh import Mesh, MeshViewers
from tensorpack import PredictConfig
from tensorpack import get_model_loader, SimpleDatasetPredictor
from tensorpack.utils import logger
from tracker.config import config as cfg

if __name__ == "__main__":
    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='../computed/fpn2mesh_eval')
    parser.add_argument('--load', nargs="+",
                        help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')

    args = parser.parse_args()
    choke_dataflow_provider = data.ChokepointDataflowProvider()
    # tracker_model = track_model.ResNetFPNTrackModel()
    coma_model = feature_mesh_model.FeatureToMeshComa(is_eval=True, edge_vertices=choke_dataflow_provider.edge_vertices)
    # Sets directory for logs, checkpoints, tensorboard
    # Keeps the directory, in case training is contiuned
    logger.set_logger_dir(args.logdir, 'k')

    print("Evaluating on Test Meshes")

    test_dataflow = choke_dataflow_provider.get_fpn_feature_mesh_val_test_dataflow()

    checkpoint_path = os.path.join(args.logdir, "checkpoint")
    stepnum = test_dataflow.size()

    session_init = get_model_loader(args.load)
    # session_init = get_model_loader("../computed/coma/tp_coma_fpn_features_to_mesh-16-16-16-32/checkpoint")
    # session_init = get_model_loader("../computed/coma/test_with_other_mean_std'")

    pred_config = PredictConfig(
        model=coma_model,
        session_init=session_init,
        input_names=["coma_fpn_feature_input", "coma_target_mesh"],
        output_names=["fpn2mesh/coma_decoder/outputs/reconstructed_mesh",
                      "coma_loss/total_loss"]
    )

    pred = SimpleDatasetPredictor(pred_config, test_dataflow)

    results = pred.get_all_result()
    predictions = []
    losses = []
    for res in results:
        preds = res[0]
        losss = res[1]
        predictions.extend(preds)
        losses.append(losss)
        # print("first_res")

    total_loss = np.sum(losses) * cfg.COMA.BATCH_SIZE / stepnum
    print("L1 Loss: {}".format(total_loss))
    predictions = np.asarray(predictions)

    euclidean_loss = np.mean(np.sqrt(
        np.sum((choke_dataflow_provider.std * (
                    predictions - choke_dataflow_provider.val_meshes[:predictions.shape[0]])) ** 2, axis=2)))

    print("euclidean loss: {}".format(euclidean_loss))
    viewer = MeshViewers(shape=(1, 2), titlebar="side by side")
    viewer[0][0].set_dynamic_meshes([Mesh(
        v=(choke_dataflow_provider.std * predictions[50] + choke_dataflow_provider.mean),
        f=choke_dataflow_provider.template_mesh.f)])
    viewer[0][1].set_dynamic_meshes([Mesh(
        v=(choke_dataflow_provider.std * choke_dataflow_provider.val_meshes[50] + choke_dataflow_provider.mean),
        f=choke_dataflow_provider.template_mesh.f)])

    print("resulteloni")
