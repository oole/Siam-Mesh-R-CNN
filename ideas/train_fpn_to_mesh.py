import argparse
import os

import tensorflow._api.v2.compat.v1 as tf
from tracker.coma_model.data import data
from tracker.model.coma import feature_mesh_model
from tensorpack import PeriodicCallback, ModelSaver, PeakMemoryTracker, EstimatedTimeLeft, \
    SessionRunTimeout, get_model_loader, QueueInput, SyncMultiGPUTrainerReplicated, \
    launch_train_with_config, InferenceRunner, ScalarStats, TrainConfig
from tensorpack.utils import logger
from tracker.config import finalize_configs, config as cfg

if __name__ == "__main__":

    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory',
                        default='../computed/coma/tp_coma_fpn_features_to_mesh-16-16-16-32-edge-loss')
    parser.add_argument('--load', nargs="+",
                        help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')

    args = parser.parse_args()

    choke_dataflow_provider = data.ChokepointDataflowProvider()
    train_dataflow = choke_dataflow_provider.get_fpn_feature_mesh_train_dataflow()
    eval_dataflow = choke_dataflow_provider.get_fpn_feature_mesh_val_dataflow()

    stepnum = train_dataflow.size()

    cfg.COMA.DECAY_STEPS = stepnum
    cfg.COMA.GRAPH_CONV_FILTERS = [16, 16, 16, 32]

    coma_model = feature_mesh_model.FeatureToMeshComa(edge_vertices=choke_dataflow_provider.edge_vertices)
    # Sets directory for logs, checkpoints, tensorboard
    # Keeps the directory, in case training is contiuned
    logger.set_logger_dir(args.logdir, 'k')

    print("training on Chokepoint images + meshes")

    finalize_configs(is_training=True)

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
                                                     ScalarStats(
                                                         ['tower0/coma_loss/data_loss/absolute_difference/value:0',
                                                          'tower0/coma_loss/total_loss:0',
                                                          'tower0/coma_loss/edge_loss/absolute_difference/value:0'],
                                                         prefix="val")),
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
            # provide list of args here:
            session_init = get_model_loader(args.load)
        else:
            # initialize with all kinds of weights
            session_init = get_model_loader(cfg.COMA.WEIGHTS) if len(cfg.COMA.WEIGHTS.to_dict().keys()) > 0 else None

    max_epoch = 900
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
