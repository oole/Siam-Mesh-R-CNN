import argparse
import os

import tensorflow._api.v2.compat.v1 as tf
from tensorpack import PeriodicCallback, ModelSaver, PeakMemoryTracker, EstimatedTimeLeft, \
    SessionRunTimeout, get_model_loader, QueueInput, SyncMultiGPUTrainerReplicated, \
    launch_train_with_config, InferenceRunner, ScalarStats, TrainConfig
from tensorpack.utils import logger

from tracker.coma_model.data import data
from tracker.config import finalize_configs, config as cfg
from tracker.model.coma import coma_autoencoder_model

if __name__ == "__main__":

    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='../computed/coma/tensorpack-test-9')
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--loss', help="If edge loss should be added for training, either 'edge' or 'l1-only",
                        default='edge')
    parser.add_argument('--conv-mode', help="The conv mode, either 'cheb' or 'spiral'", default="spiral")

    args = parser.parse_args()

    dataflow_provider = data.ChokepointDataflowProvider(mesh_path=cfg.COMA.TEMPLATE_MESH_TRACKER)
    train_dataflow = dataflow_provider.get_mesh_mesh_train_dataflow()
    eval_dataflow = dataflow_provider.get_mesh_mesh_val_dataflow()

    stepnum = train_dataflow.size()

    cfg.COMA.GRAPH_CONV_FILTERS = [16, 16, 16, 32]
    cfg.COMA.DOWNSAMPLING_FACTORS = [4, 4, 4, 2]

    # tracker_model = track_model.ResNetFPNTrackModel()
    if args.loss == "edge":
        coma_model = coma_autoencoder_model.Coma(edge_vertices=dataflow_provider.edge_vertices,
                                                 conv_mode=args.conv_mode, mesh_path=cfg.COMA.TEMPLATE_MESH_TRACKER)
    elif args.loss == "l1-only":
        coma_model = coma_autoencoder_model.Coma(conv_mode=args.conv_mode, mesh_path=cfg.COMA.TEMPLATE_MESH_TRACKER)
    else:
        raise Exception("Unknown loss mode, must be 'edge' or 'l1-only!")

    # Sets directory for logs, checkpoints, tensorboard
    # Keeps the directory, in case training is contiuned
    logger.set_logger_dir(args.logdir, 'k')

    finalize_configs(is_training=True)

    if args.loss == 'edge':
        scalar_stats = ScalarStats(
            ['tower0/coma_loss/data_loss/absolute_difference/value:0',
             'tower0/coma_loss/total_loss:0',
             'tower0/coma_loss/edge_loss/absolute_difference/value:0'], prefix="val")
    elif args.loss == 'l1-only':
        scalar_stats = ScalarStats(
            ['tower0/coma_loss/data_loss/absolute_difference/value:0',
             'tower0/coma_loss/total_loss:0'], prefix="val")
    else:
        raise Exception("Unknown loss mode, must be 'edge' or 'l1-only'")

    callbacks = [
                    PeriodicCallback(
                        ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                        # every_k_epochs=1),
                        every_k_epochs=10),
                    # linear warmup
                    PeakMemoryTracker(),
                    EstimatedTimeLeft(median=True),
                    SessionRunTimeout(60000).set_chief_only(True),  # 1 minute timeout
                ] + [
                    PeriodicCallback(InferenceRunner(eval_dataflow,
                                                     scalar_stats),
                                     every_k_epochs=10)
                    # eval_util.EvalCallback(eval_dataflow, *coma_model.get_inference_tensor_names(), args.logdir)
                ]

    # GPUUtilizationTracker not working with TF2 backend
    # callbacks.append(GPUUtilizationTracker())
    start_epoch = cfg.TRAIN.STARTING_EPOCH
    # first try to find existing model

    checkpoint_path = os.path.join(args.logdir, "checkpoint")
    if os.path.exists(checkpoint_path):
        session_init = get_model_loader(checkpoint_path)
        start_step = int(session_init.path.split("-")[-1])
        start_epoch = start_step // stepnum
        logger.info(
            "initializing from existing model, " + session_init.path + ", starting from epoch " + str(start_epoch))
    else:
        if args.load:
            session_init = get_model_loader(args.load)
        else:
            # initialize with all kinds of weights
            session_init = get_model_loader(cfg.COMA.WEIGHTS) if len(cfg.COMA.WEIGHTS.to_dict().keys()) > 0 else None

    max_epoch = 300
    traincfg = TrainConfig(
        model=coma_model,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        # max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        max_epoch=max_epoch,
        session_init=session_init,
        starting_epoch=start_epoch,
    )
    # nccl mode appears faster than cpu mode
    # Use Normal trainer...
    trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    launch_train_with_config(traincfg, trainer)
    print("got model")
