
# -*- coding: utf-8 -*-
# File: eval.py

import itertools
import json
import os
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import cv2
import numpy as np
import tensorflow._api.v2.compat as tf
import tqdm
from tensorpack.callbacks import Callback
from tensorpack import Inferencer
from tensorpack.tfutils.common import get_tf_version_tuple, get_op_tensor_name
from tensorpack.utils import logger
from tensorpack.utils.utils import get_tqdm

from tracker.config import config as cfg
from tracker.data.data import get_eval_dataflow
from tracker.data.dataset import DetectionDataset
from tracker.util.data_util import CustomResize, clip_boxes, box_to_point8, point8_to_box
from tracker.util import eval_util as tracker_eval_util

def predict_mesh_reconstruction(input_mesh, target_mesh, model_func):
    reconstructed_mesh = model_func(input_mesh)
    absolute_loss = reconstructed_mesh-target_mesh
    return absolute_loss

def predict_dataflow(df, model_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        model_func: a callable from the TF model.
            It takes image and returns (boxes, probs, labels, [masks])
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, in the format used by
        `DetectionDataset.eval_or_save_inference_results`
    """
    df.reset_state()
    all_results = []
    with ExitStack() as stack:
        # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(get_tqdm(total=df.size()))
        for input_mesh, target_mesh in df:
            absolute_loss = predict_mesh_reconstruction(input_mesh, target_mesh, model_func)
            all_results.append(absolute_loss)
            tqdm_bar.update(1)
        # TODO mesh std
        euclidean_loss = np.mean(np.sqrt(np.sum(np.asarray(all_results)**2, axis=2)))
    return euclidean_loss

class EvalCallback(Callback):
    """
    A callback that runs evaluation once a while.
    It supports multi-gpu evaluation.
    """

    _chief_only = False

    def __init__(self, eval_dataflow, in_names, out_names, output_dir):
        self.eval_dataflow = eval_dataflow
        self._in_names, self._out_names = in_names, out_names
        self._output_dir = output_dir

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        # TF bug in version 1.11, 1.12: https://github.com/tensorflow/tensorflow/issues/22750
        buggy_tf = get_tf_version_tuple() in [(1, 11), (1, 12)]

        # Only use one thread for one dataflow
        self.num_predictor = num_gpu
        self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]

    def _build_predictor(self, idx):
        return self.trainer.get_predictor(self._in_names, self._out_names, device=idx)

    def _before_train(self):
        eval_period = cfg.COMA.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in itertools.count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every {} epochs".format(eval_period))

    def _eval(self):
        logdir = self._output_dir
        euclidean_loss = predict_dataflow(self.eval_dataflow, self.predictors[0])

        output_file = os.path.join(
            logdir, '{}-_outputs{}.json'.format("Eval_MeshReconstruction", self.global_step))

        # scores = DetectionDataset().eval_or_save_inference_results(
        #     all_results, self._eval_dataset, output_file)

        self.trainer.monitors.put_scalar("mesh_reconstruction_val", euclidean_loss)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()

class ScalarStats(Inferencer):
    """
    Statistics of some scalar tensor.
    The value will be averaged over all given datapoints.

    Note that the average of accuracy over all batches is not necessarily the
    accuracy of the whole dataset. See :class:`ClassificationError` for details.
    """

    def __init__(self, names, prefix='validation'):
        """
        Args:
            names(list or str): list of names or just one name. The
                corresponding tensors have to be scalar.
            prefix(str): a prefix for logging
        """
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names
        self.prefix = prefix

    def _before_inference(self):
        self.stats = []

    def _get_fetches(self):
        return self.names

    def _on_fetches(self, output):
        self.stats.append(output)

    def _after_inference(self):
        if len(self.stats):
            self.stats = np.mean(self.stats, axis=0)
            assert len(self.stats) == len(self.names)

        ret = {}
        for stat, name in zip(self.stats, self.names):
            opname, _ = get_op_tensor_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            ret[name] = stat
        return ret