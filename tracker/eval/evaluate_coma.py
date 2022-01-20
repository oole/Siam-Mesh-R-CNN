import argparse

import tensorflow._api.v2.compat.v1 as tf
from tensorpack import get_model_loader, PredictConfig, SimpleDatasetPredictor, OfflinePredictor
from tensorpack.utils import logger
from tracker.config import config as cfg
from tracker.model.coma import coma_autoencoder_model
from tracker.coma_model.data import data
import numpy as np
from psbody.mesh import Mesh, MeshViewers

if __name__ == "__main__":

    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='../../computed/coma/tensorpack-test-9-coma-eval')
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--loss', help="If edge loss should be added for training, either 'edge' or 'l1-only",
                        default='edge')
    parser.add_argument('--conv-mode', help="The conv mode, either 'cheb' or 'spiral'", default="spiral")
    parser.add_argument('--result-dir', help='The results directory to which the tests should be run',
                        default="results_coma")
    parser.add_argument('--eval-name', help="The name under which the results should be stored", required=True)
    parser.add_argument('--visualize', default=False)

    args = parser.parse_args()

    ## TODO Add test dataflow
    dataflow_provider = data.ChokepointDataflowProvider(mesh_path=cfg.COMA.TEMPLATE_MESH_EVAL)
    test_dataflow = dataflow_provider.get_mesh_mesh_val_dataflow()

    cfg.COMA.GRAPH_CONV_FILTERS = [16, 16, 16, 32]
    cfg.COMA.DOWNSAMPLING_FACTORS = [4, 4, 4, 2]

    # tracker_model = track_model.ResNetFPNTrackModel()
    if args.loss == "edge":
        coma_model = coma_autoencoder_model.Coma(edge_vertices=dataflow_provider.edge_vertices,
                                                 conv_mode=args.conv_mode, is_eval=True, mesh_path=cfg.COMA.TEMPLATE_MESH_EVAL)
    elif args.loss == "l1-only":
        coma_model = coma_autoencoder_model.Coma(conv_mode=args.conv_mode, is_eval=True, mesh_path=cfg.COMA.TEMPLATE_MESH_EVAL)
    else:
        raise Exception("Unknown loss mode, must be 'edge' or 'l1-only!")

    # Sets directory for logs, checkpoints, tensorboard
    # Keeps the directory, in case training is contiuned
    logger.set_logger_dir(args.logdir, 'k')

    stepnum = test_dataflow.size()

    session_init = get_model_loader(args.load)

    pred_config = PredictConfig(
        model=coma_model,
        session_init=session_init,
        input_names=["coma_input_mesh", "coma_target_mesh"],
        output_names=["mesh2feature/coma_decoder/outputs/reconstructed_mesh",
                      "coma_loss/total_loss"]
    )

    perform_classic_val = True
    if perform_classic_val:
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
            np.sum((dataflow_provider.std * (
                    predictions - dataflow_provider.val_meshes[:predictions.shape[0]])) ** 2, axis=2)))

        print("euclidean loss: {}".format(euclidean_loss))

        np.save(args.result_dir + "/" + args.eval_name + "_result", predictions)


        if args.visualize:
            viewer = MeshViewers(shape=(1, 2), titlebar="side by side")
            def show_mesh(viewer, number):
                viewer[0][0].set_dynamic_meshes([Mesh(
                    v=(dataflow_provider.std * predictions[number] + dataflow_provider.mean),
                    f=dataflow_provider.template_mesh.f)])
                viewer[0][1].set_dynamic_meshes([Mesh(
                    v=(dataflow_provider.std * dataflow_provider.val_meshes[number] + dataflow_provider.mean),
                    f=dataflow_provider.template_mesh.f)])

            show_mesh(viewer, 1400)
        print("Done.")
    else:
        ###  get single result for sanity check:
        extract_func = OfflinePredictor(pred_config)

        viewer = MeshViewers(shape=(1, 2), titlebar="side by side")


        def show_mesh(viewer, mesh, index):
            viewer[0][0].set_dynamic_meshes([Mesh(
                v=(dataflow_provider.std * mesh + dataflow_provider.mean),
                f=dataflow_provider.template_mesh.f)])
            viewer[0][1].set_dynamic_meshes([Mesh(
                v=(dataflow_provider.std * dataflow_provider.val_meshes[index] + dataflow_provider.mean),
                f=dataflow_provider.template_mesh.f)])

        def get_result_for_index(index):
            result = extract_func(np.expand_dims(dataflow_provider.val_meshes[index], axis=0),
                              np.expand_dims(dataflow_provider.val_meshes[index], axis=0))
            mesh = result[0][0]
            show_mesh(viewer, mesh, index)

        get_result_for_index(1400)

    print("resulteloni")