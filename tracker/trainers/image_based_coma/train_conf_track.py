import argparse
import os

import tensorflow._api.v2.compat.v1 as tf
from tensorpack import PeriodicCallback, ModelSaver, ScheduledHyperParamSetter, PeakMemoryTracker, EstimatedTimeLeft, \
    SessionRunTimeout, get_model_loader, TrainConfig, QueueInput, SyncMultiGPUTrainerReplicated, \
    launch_train_with_config
from tensorpack.utils import logger

from tracker.config import finalize_configs, config as cfg
from tracker.data import data
from tracker.data.data import DetectionDataset
from tracker.model.leftovers import conv_track_model
from tracker.util import eval_util

if __name__ == "__main__":
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='../computed/test_conv_track_0')
    parser.add_argument('--load', nargs="+", help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')


    args = parser.parse_args()


    # tracker_model = track_model.ResNetFPNTrackModel()
    tracker_model = conv_track_model.ConvTrackModel()
    # Sets directory for logs, checkpoints, tensorboard
    # Keeps the directory, in case training is contiuned
    logger.set_logger_dir(args.logdir, 'k')

    # initialize config with dataset
    DetectionDataset()

    finalize_configs(is_training=True)

    stepnum = 1  # cfg.TRAIN.STEPS_PER_EPOCH

    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))

    train_dataflow = data.get_train_dataflow()

    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()

    callbacks = [
                    PeriodicCallback(
                        ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                        # every_k_epochs=1),
                        every_k_epochs=20),
                    # linear warmup
                    ScheduledHyperParamSetter(
                        'learning_rate', warmup_schedule, interp='linear', step_based=True),
                    ScheduledHyperParamSetter('learning_rate', lr_schedule),
                    PeakMemoryTracker(),
                    EstimatedTimeLeft(median=True),
                    SessionRunTimeout(60000).set_chief_only(True),  # 1 minute timeout
                ] + [
                    eval_util.EvalCallback(dataset, *tracker_model.get_inference_tensor_names(), args.logdir)
                    for dataset in cfg.DATA.VAL
                ]

    #GPUUtilizationTracker not working with TF2 backend
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
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    max_epoch = min(cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum, cfg.TRAIN.MAX_NUM_EPOCHS)

    traincfg = TrainConfig(
        model=tracker_model,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        # max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        max_epoch=max_epoch,
        session_init=session_init,
        starting_epoch=start_epoch
    )
    # nccl mode appears faster than cpu mode
    # Use Normal trainer...
    trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    launch_train_with_config(traincfg, trainer)
    print("got model")
