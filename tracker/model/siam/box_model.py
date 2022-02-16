from collections import namedtuple

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from tensorpack.tfutils.scope_utils import under_name_scope
from tracker.model.coma import mesh_model
from tracker.config import config as cfg

@under_name_scope()
def roi_align(featuremap, featuremap_boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution

    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, featuremap_boxes,
        tf.zeros([tf.shape(featuremap_boxes)[0]], dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret

@under_name_scope()
def roi_align_with_meshes(image, image_boxes, featuremap, featuremap_boxes, resolution, mesh_helper):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution

    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, featuremap_boxes,
        tf.zeros([tf.shape(featuremap_boxes)[0]], dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    meshes = get_mesh_features(
        image, image_boxes,
        tf.zeros([tf.shape(image_boxes)[0]], dtype=tf.int32), 224, mesh_helper)
    ret = tf.concat((ret, meshes), axis=1)
    return ret


@under_name_scope()
def get_mesh_features(image, boxes, box_ind, crop_size, mesh_helper: mesh_model.MeshModel, pad_border=True):
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        # pad w and h of image
        image = tf.pad(image, [[0, 0], [0, 0], [crop_size, crop_size], [crop_size, crop_size]], mode='SYMMETRIC')
        # expand boxes according to padding
        boxes = boxes + crop_size

    @under_name_scope()
    def transform_bboxes_for_crop_on_image(boxes, image_shape, crop_shape):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)  # width / crop width
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32) # height / crop height

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]  # shape[0] = 600+crop size, shape[1] = 800 + crop size
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)


    image_shape = tf.shape(image)[2:]
    boxes = transform_bboxes_for_crop_on_image(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])  # nhwc
    crops = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])

    ## now get Mesh features:
    mesh_features = mesh_helper.img2features(crops)

    mesh_features = tf.reshape(mesh_features, shape=(-1, 7, 7))
    mesh_features = tf.expand_dims(mesh_features, axis=1)
    return mesh_features


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32
        anchors: (..., 4), float32

    Returns:
        box_encoded: (..., 4), float32 with the same shape.
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))


@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = np.log(cfg.PREPROC.MAX_SIZE / 16.)  # tf func?
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5  # (...)x1x2
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        # pad w and h of image
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        # expand boxes according to padding
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])  # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])  # ncss
    return ret


@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])  # (4,)
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes


class RPNAnchors(namedtuple('_RPNAnchors', ['boxes', 'gt_labels', 'gt_boxes'])):
    """
    boxes (FS x FS x NA x 4): The anchor boxes.
    gt_labels (FS x FS x NA):
    gt_boxes (FS x FS x NA x 4): Groundtruth boxes corresponding to each anchor.
    """

    def encoded_gt_boxes(self):
        return encode_bbox_target(self.gt_boxes, self.boxes)

    def decode_logits(self, logits):
        return decode_bbox_target(logits, self.boxes)

    @under_name_scope()
    def narrow_to(self, featuremap):
        """
        Slice anchors to the spatial size of this featuremap.
        """
        shape2d = tf.shape(featuremap)[2:]  # h,w
        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)
        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        gt_labels = tf.slice(self.gt_labels, [0, 0, 0], slice3d)
        gt_boxes = tf.slice(self.gt_boxes, [0, 0, 0, 0], slice4d)
        return RPNAnchors(boxes, gt_labels, gt_boxes)
