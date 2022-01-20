# -*- coding: utf-8 -*-
# File: data.py

import bisect
import glob
import os
import random

import PIL
import cv2
import numpy as np
import xmltodict
from tabulate import tabulate
from tensorpack.dataflow import (
    DataFromList, MultiThreadMapData, MapData, imgaug)
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once, memoized
from termcolor import colored

from tracker.config import config as cfg
from tracker.data.dataset import DetectionDataset
from tracker.util.anchor_util import generate_anchors
from tracker.util.data_util import (
    CustomResize, box_to_point8,
    filter_boxes_inside_shape, point8_to_box, np_iou)
from tracker.util.hard_example_utils import subsample_nns
from tracker.util.np_box_ops import area as np_area, ioa as np_ioa


# import tensorpack.utils.viz as tpviz


class MalformedData(BaseException):
    pass


def print_class_histogram(roidbs):
    """
    Args:
        roidbs (list[dict]): the same format as the output of `load_training_roidbs`.
    """
    dataset = DetectionDataset()
    hist_bins = np.arange(dataset.num_classes + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((dataset.num_classes,), dtype=np.int)
    for entry in roidbs:
        # filter crowd?
        gt_inds = np.where(
            (entry['class'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['class'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    data = [[dataset.class_names[i], v] for i, v in enumerate(gt_hist)]
    data.append(['total', sum([x[1] for x in data])])
    table = tabulate(data, headers=['class', '#box'], tablefmt='pipe')
    logger.info("Ground-Truth Boxes:\n" + colored(table, 'cyan'))


@memoized
def get_all_anchors(stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = cfg.PREPROC.MAX_SIZE
    field_size = int(np.ceil(max_size / stride))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
            cell_anchors.reshape((1, A, 4)) +
            shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors


@memoized
def get_all_anchors_fpn(strides=None, sizes=None):
    """
    Returns:
        [anchors]: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array.
    """
    if strides is None:
        strides = cfg.FPN.ANCHOR_STRIDES
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,))
        foas.append(foa)
    return foas


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float, non-crowd
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """

    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1  # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')  # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= cfg.RPN.POSITIVE_ANCHOR_THRESH] = 1
    anchor_labels[ious_max_per_anchor < cfg.RPN.NEGATIVE_ANCHOR_THRESH] = 0

    # label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        ioas = np_ioa(crowd_boxes, cand_anchors)
        overlap_with_crowd = cand_inds[ioas.max(axis=0) > cfg.RPN.CROWD_OVERLAP_THRESH]
        anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(cfg.RPN.BATCH_PER_IM * cfg.RPN.FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Keep an image even if there is no foreground anchors
    # if len(fg_inds) == 0:
    #     raise MalformedData("No valid foreground for RPN!")

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg in this image, skip.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = cfg.RPN.BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)  # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.RPN.BATCH_PER_IM
    return anchor_labels, anchor_boxes


def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
        NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    boxes = boxes.copy()
    all_anchors = np.copy(get_all_anchors())
    # fHxfWxAx4 -> (-1, 4)
    featuremap_anchors_flatten = all_anchors.reshape((-1, 4))

    # only use anchors inside the image
    inside_ind, inside_anchors = filter_boxes_inside_shape(featuremap_anchors_flatten, im.shape[:2])
    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # Fill them back to original size: fHxfWx1, fHxfWx4
    anchorH, anchorW = all_anchors.shape[:2]
    featuremap_labels = -np.ones((anchorH * anchorW * cfg.RPN.NUM_ANCHOR,), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR))
    featuremap_boxes = np.zeros((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_gt_boxes
    featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes


def get_multilevel_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level.
        Each tuple contains the anchor labels and target boxes for each pixel in the featuremap.

        fm_labels: fHxfWx NUM_ANCHOR_RATIOS
        fm_boxes: fHxfWx NUM_ANCHOR_RATIOS x4
    """
    boxes = boxes.copy()
    anchors_per_level = get_all_anchors_fpn()
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)

    inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, im.shape[:2])
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors,), dtype='int32')
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype='float32')
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        assert level_anchor.shape[2] == len(cfg.RPN.ANCHOR_RATIOS)
        anchor_shape = level_anchor.shape[:3]  # fHxfWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = np.prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append(
            (all_labels[start: end].reshape(anchor_shape),
             all_boxes[start: end, :].reshape(anchor_shape + (4,))
             ))
        start = end
    assert end == num_all_anchors, "{} != {}".format(end, num_all_anchors)
    return multilevel_inputs


def get_bbox_from_segmentation_mask_np(mask):
    object_locations = (np.stack(np.where(np.equal(mask, 1))).T[:, :2]).astype(np.int32)
    y0 = np.min(object_locations[:, 0])
    x0 = np.min(object_locations[:, 1])
    y1 = np.max(object_locations[:, 0]) + 1
    x1 = np.max(object_locations[:, 1]) + 1
    bbox = np.stack([x0, y0, x1, y1])
    return bbox


def _augment_boxes(boxes, aug, params):
    points = box_to_point8(boxes)
    points = aug.augment_coords(points, params)
    boxes = point8_to_box(points)
    # assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"
    if np.min(np_area(boxes)) <= 0:
        return None
    return boxes


def _preprocess_common(ref_box, target_box, ref_im, target_im, aug):
    ref_boxes = np.array([ref_box], dtype=np.float32)
    target_boxes = np.array([target_box], dtype=np.float32)
    klass = np.array([1], dtype=np.int32)

    # augmentation:
    target_im, target_params = aug.augment_return_params(target_im)
    ref_im, ref_params = aug.augment_return_params(ref_im)
    ref_boxes = _augment_boxes(ref_boxes, aug, ref_params)
    target_boxes = _augment_boxes(target_boxes, aug, target_params)
    if ref_boxes is None or target_boxes is None:
        return None

    # additional augmentations:
    # motion blur
    if cfg.DATA.MOTION_BLUR_AUGMENTATIONS:
        do_motion_blur_ref = np.random.rand() < 0.25
        if do_motion_blur_ref:
            # generating the kernel
            kernel_size = np.random.randint(5, 15)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            # applying the kernel
            ref_im = cv2.filter2D(ref_im, -1, kernel_motion_blur)
        do_motion_blur_target = np.random.rand() < 0.25
        if do_motion_blur_target:
            # generating the kernel
            kernel_size = np.random.randint(5, 15)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            # applying the kernel
            target_im = cv2.filter2D(target_im, -1, kernel_motion_blur)

    # grayscale
    if cfg.DATA.GRAYSCALE_AUGMENTATIONS:
        do_grayscale = np.random.rand() < 0.25
        if do_grayscale:
            grayscale_aug = imgaug.Grayscale()
            ref_im = np.tile(grayscale_aug.augment(ref_im), [1, 1, 3])
            target_im = np.tile(grayscale_aug.augment(target_im), [1, 1, 3])

    if cfg.DATA.DEBUG_VIS:
        import matplotlib.pyplot as plt
        ref_im_vis = ref_im.copy()
        # ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]), 0] = 255
        ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]), 2] = \
            (0.5 * ref_im_vis[int(ref_boxes[0][1]):int(ref_boxes[0][3]), int(ref_boxes[0][0]):int(ref_boxes[0][2]),
                   2] + 120).astype(np.uint8)
        plt.imshow(ref_im_vis[..., ::-1])
        plt.show()
        target_im_vis = target_im.copy()
        target_im_vis[int(target_boxes[0][1]):int(target_boxes[0][3]), int(target_boxes[0][0]):int(target_boxes[0][2]),
        2] = \
            (0.5 * target_im_vis[int(target_boxes[0][1]):int(target_boxes[0][3]),
                   int(target_boxes[0][0]):int(target_boxes[0][2]), 2] + 120).astype(np.uint8)
        plt.imshow(target_im_vis[..., ::-1])
        plt.show()

    is_crowd = np.array([0], dtype=np.int32)
    ret = {'ref_image': ref_im, 'ref_box': ref_boxes[0], 'image': target_im}
    if cfg.DATA.DEBUG_VIS:
        return ret

    # rpn anchor:
    try:
        if cfg.MODE_FPN:
            multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(target_im, target_boxes, is_crowd)
            for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
                ret['anchor_labels_lvl{}'.format(i + 2)] = anchor_labels
                ret['anchor_boxes_lvl{}'.format(i + 2)] = anchor_boxes
        else:
            # anchor_labels, anchor_boxes
            ret['anchor_labels'], ret['anchor_boxes'] = get_rpn_anchor_input(target_im, target_boxes, is_crowd)
        ret['gt_boxes'] = target_boxes
        ret['gt_labels'] = klass
        if not len(target_boxes):
            raise MalformedData("No valid gt_boxes!")
    except MalformedData as e:
        log_once("Input is filtered for training: {}".format(str(e)), 'warn')
        return None
    return ret


def _preprocess_chokepoint(roidb, aug):
    vid_name = roidb
    ann_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation/G1/train", vid_name)
    ann_files = sorted(glob.glob(ann_path + "/*.xml"))
    # randomly select two files
    ref_idx = np.random.randint(len(ann_files))
    target_idx = np.random.randint(1, len(ann_files))
    ref_ann_file = ann_files[ref_idx]
    target_ann_file = ann_files[target_idx]

    def get_id_to_data(ann):
        id_to_data = {}
        if "object" in ann:
            obj_anns = ann["object"]
            if not isinstance(obj_anns, list):
                obj_anns = [obj_anns]
            for obj_ann in obj_anns:
                id_ = obj_ann["trackid"]
                id_to_data[id_] = obj_ann
        return id_to_data

    ref_ann = xmltodict.parse(open(ref_ann_file).read())["annotation"]
    target_ann = xmltodict.parse(open(target_ann_file).read())["annotation"]
    ref_id_to_data = get_id_to_data(ref_ann)
    target_id_to_data = get_id_to_data(target_ann)
    ref_obj_ids = set(ref_id_to_data.keys())
    target_obj_ids = set(target_id_to_data.keys())
    obj_ids = ref_obj_ids & target_obj_ids
    obj_ids = list(obj_ids)
    if len(obj_ids) == 0:
        # this happens quite often, do not print it for now
        # log_once("Inputs {},{} filtered for training because of no common objects".format(ref_fname, target_fname),
        #         'warn')
        return None
    random.shuffle(obj_ids)
    obj_id = obj_ids[0]

    def obj_data_to_bbox(obj_ann):
        bbox = obj_ann['bndbox']
        x1 = bbox['xmin']
        y1 = bbox['ymin']
        x2 = bbox['xmax']
        y2 = bbox['ymax']
        box = [x1, y1, x2, y2]
        return box

    def obj_file_path(obj_ann):
        sub_path = obj_ann['folder']
        fname = obj_ann['filename']
        full_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, sub_path, fname + ".jpg")
        return full_path

    ref_fname = obj_file_path(ref_ann)
    target_fname = obj_file_path(target_ann)

    ref_ann = ref_id_to_data[obj_id]
    target_ann = target_id_to_data[obj_id]
    ref_box = obj_data_to_bbox(ref_ann)
    target_box = obj_data_to_bbox(target_ann)

    ref_im = cv2.imread(ref_fname, cv2.IMREAD_COLOR)

    target_im = cv2.imread(target_fname, cv2.IMREAD_COLOR)
    data = _preprocess_common(ref_box, target_box, ref_im, target_im, aug)
    vid_name = roidb.replace("/", "_") + "_" + str(obj_id)
    return data



def get_train_dataflow():
    roidbs = DetectionDataset().load_training_roidbs(cfg.DATA.TRAIN)
    ds = DataFromList(roidbs, shuffle=True)
    # for now let's not do flipping to keep things simple
    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])  # ,
    # imgaug.Flip(horiz=True)])

    hard_mining_index = None
    hard_mining_names = None

    def preprocess(roidb):
        if roidb.startswith("P"):
            return _preprocess_chokepoint(roidb, aug)
        else:
            assert False

    # ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    # ds = MapData(ds, preprocess)
    if cfg.DATA.DEBUG_VIS or not cfg.DATA.MULTITHREAD:
        ds = MapData(ds, preprocess)
    else:
        # ds = MultiThreadMapData(ds, 6, preprocess)
        ds = MultiThreadMapData(ds, 8, preprocess, buffer_size=80)
    return ds


def get_eval_dataflow(name, shard=0, num_shards=1):
    seqs = []
    # with open("davis2017_fast_val_ids.txt") as f:
    #     for l in f:
    #         seqs.append(l.strip())
    subset_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation", "G1", name)
    paths = sorted(glob.glob(subset_path + "*/*/*/*"))
    sequences_1 = [path for path in paths if "xml" not in path and "seq" in path]
    sequences_1 = ["/".join(v.split("/")[-3:]) for v in sequences_1]
    paths = sorted(glob.glob(subset_path + "*/*/*"))
    sequences_2 = [path for path in paths if "xml" not in path and "seq" in path]
    sequences_2 = ["/".join(v.split("/")[-2:]) for v in sequences_2]
    sequences = list(sequences_2 + sequences_1)
    # use subset:
    random.Random(4).shuffle(sequences)
    sequences = sequences[:10]

    seqs_timesteps = []
    for seq in sequences:
        sequence_folder = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation/G1/", name, seq)
        sequence_frame_files = sorted(glob.glob(sequence_folder + "/*.xml"))
        sequence_frame_numbers = [frame_file.split("/")[-1].split(".")[0] for frame_file in sequence_frame_files]
        first_frame_number = sequence_frame_numbers[0]
        for timestep in sequence_frame_numbers:
            seqs_timesteps.append((sequence_folder, timestep, first_frame_number))

    num_seqs_timesteps = len(seqs_timesteps)
    seqs_timesteps_per_shard = num_seqs_timesteps // num_shards
    seqs_timesteps_range = (shard * seqs_timesteps_per_shard,
                            (shard + 1) * seqs_timesteps_per_shard if shard + 1 < num_shards else num_seqs_timesteps)
    ds = DataFromList(seqs_timesteps[seqs_timesteps_range[0]: seqs_timesteps_range[1]])

    def preprocess(seq_timestep):
        # ff_fn is the first frame of the sequence
        # ff_box is the bounding box for the first frame
        # -> get that from the sequence folder
        # -> that's the ref_bb
        # ann_fn is the frame that is evaluated
        # that is what we're getting from the timestep.
        # ann_bbox is the frame that is being evaluated on ann_fn from the current timestep
        # ->> thats the target bbox that should be found by the detector
        sequence_annotation_dir, target_frame, first_frame = seq_timestep
        target_frame_annotation = os.path.join(sequence_annotation_dir, target_frame + ".xml")
        first_frame_annotation = os.path.join(sequence_annotation_dir, first_frame + ".xml")

        target_frame_dict = xmltodict.parse(open(target_frame_annotation).read())["annotation"]
        first_frame_dict = xmltodict.parse(open(first_frame_annotation).read())["annotation"]

        def obj_data_to_bbox(annotation):
            object_ann = annotation['object']
            bbox = object_ann['bndbox']
            x1 = bbox['xmin']
            y1 = bbox['ymin']
            x2 = bbox['xmax']
            y2 = bbox['ymax']
            box = [x1, y1, x2, y2]
            return box

        target_frame_bbox = np.array(obj_data_to_bbox(target_frame_dict), dtype=np.float32)
        first_frame_bbox = np.array(obj_data_to_bbox(first_frame_dict), dtype=np.float32)

        def obj_file_path(annotation):
            sub_path = annotation['folder']
            fname = annotation['filename']
            full_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, sub_path, fname + ".jpg")
            return full_path

        def get_name(annotation, sequence_annotation_dir, target_frame):
            name = annotation['object']['name']
            frame_name = "__".join((name, sequence_annotation_dir, target_frame))
            return frame_name

        target_frame_file = obj_file_path(target_frame_dict)
        first_frame_file = obj_file_path(first_frame_dict)
        target_img = cv2.imread(target_frame_file, cv2.IMREAD_COLOR)
        ref_img = cv2.imread(first_frame_file, cv2.IMREAD_COLOR)
        return ref_img, first_frame_bbox, target_img, target_frame_bbox, get_name(target_frame_dict,
                                                                                  sequence_annotation_dir,
                                                                                  target_frame)

    ds = MapData(ds, preprocess)
    return ds
