import tensorflow._api.v2.compat.v1 as tf
from tensorpack import argscope, Conv2D, FullyConnected, regularize_cost, l2_regularizer
from tensorpack.tfutils.common import get_tensors_by_names  # No probem it's there.
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables

from tracker.config import config as cfg
from tracker.data import data
from tracker.model.coma import mesh_model
from tracker.model.siam import base_model, box_model, cascade_model, fpn_model, frcnn_model, rpn_model
from tracker.model.siam.track_model import DetectionModel
from tracker.util import common_util

tf.disable_eager_execution()

class ResNetFPNMeshModel(DetectionModel):

    def __init__(self, mesh_path=None, fpn_conv_mode="spiral", coma_conv_mode="spiral"):
        self.mesh_helper = mesh_model.MeshModel(mesh_path=mesh_path,
                                                fpn_conv_mode=fpn_conv_mode,
                                                coma_conv_mode=coma_conv_mode)

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (600, 800, 3), 'image')]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, num_anchors),
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')])  # all > 0
        if cfg.EXTRACT_GT_FEATURES:
            ret.append(tf.placeholder(tf.float32, (None, 4,), 'roi_boxes'))
        return ret

    def slice_feature_and_anchors(self, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_slice_lvl{}'.format(i)):
                anchors[i] = anchors[i].narrow_to(p23456[i])

    def backbone(self, image):
        c2345 = base_model.resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        p23456 = fpn_model.fpn_model('fpn', c2345)
        return p23456

    def rpn(self, image, features, inputs):
        if cfg.EXTRACT_GT_FEATURES:
            boxes = inputs['roi_boxes']
            return frcnn_model.BoxProposals(boxes), tf.constant(0, dtype=tf.float32)

        assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

        image_shape2d = tf.shape(image)[2:]  # h,w
        all_anchors_fpn = data.get_all_anchors_fpn()
        multilevel_anchors = [box_model.RPNAnchors(
            all_anchors_fpn[i],
            inputs['anchor_labels_lvl{}'.format(i + 2)],
            inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]
        self.slice_feature_and_anchors(features, multilevel_anchors)

        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_model.rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
                       for pi in features]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]
        multilevel_pred_boxes = [anchor.decode_logits(logits)
                                 for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]

        proposal_boxes, proposal_scores = fpn_model.generate_fpn_proposals(
            multilevel_pred_boxes, multilevel_label_logits, image_shape2d)

        if self.training:
            losses = fpn_model.multilevel_rpn_losses(
                multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
        else:
            losses = []

        return frcnn_model.BoxProposals(proposal_boxes), losses

    def get_mesh_features(self, roi_aligned_features):
        ## now get Mesh features:
        mesh_features = self.mesh_helper.fpn2features(roi_aligned_features)
        mesh_features = tf.reshape(mesh_features, shape=(-1, 7, 7))
        mesh_features = tf.expand_dims(mesh_features, axis=1)
        return mesh_features

    def roi_heads(self, image, features, proposals, targets):
        image_shape2d = tf.shape(image)[2:]  # h,w
        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        if self.training:
            proposals = frcnn_model.sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)

        fastrcnn_head_func = getattr(frcnn_model, cfg.FPN.FRCNN_HEAD_FUNC)
        if not cfg.FPN.CASCADE:
            roi_feature_fastrcnn = fpn_model.multilevel_roi_align(features[:4], proposals.boxes, 7)

            head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
            fastrcnn_label_logits, fastrcnn_box_logits = frcnn_model.fastrcnn_outputs(
                'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS)
            fastrcnn_head = frcnn_model.FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                                     gt_boxes,
                                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
        else:
            def roi_func(boxes):
                aligned_features = fpn_model.multilevel_roi_align(features[:4], boxes, 7)
                target_mesh_features = self.get_mesh_features(aligned_features)
                aligned_features = tf.concat((aligned_features, target_mesh_features), axis=1)
                return aligned_features

            fastrcnn_head = cascade_model.CascadeRCNNHead(
                proposals, roi_func, fastrcnn_head_func,
                (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS)

        if cfg.EXTRACT_GT_FEATURES:
            roi_feature_fastrcnn = fpn_model.multilevel_roi_align(features[:4], proposals.boxes, 7)
            roi_feature_fastrccn_mesh = self.get_mesh_features(roi_feature_fastrcnn)
            tf.identity(roi_feature_fastrcnn, "rpn/feature")
            tf.identity(roi_feature_fastrccn_mesh, "rpn/mesh_feature")

        if self.training:
            all_losses = fastrcnn_head.losses()

            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = box_model.clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = frcnn_model.fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')
            return []


class MeshTrackModel(ResNetFPNMeshModel):
    def __init__(self, mesh_path=None, fpn_conv_mode="spiral", coma_conv_mode="spiral"):
        super().__init__(mesh_path=mesh_path, fpn_conv_mode=fpn_conv_mode, coma_conv_mode=coma_conv_mode)

    def inputs(self):
        ret = super().inputs()
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            ret.append(tf.placeholder(tf.float32, (256, 7, 7), 'ref_features'))
            ret.append(tf.placeholder(tf.float32, (1, 7, 7), 'ref_mesh_features'))
        else:
            ret.append(tf.placeholder(tf.float32, (600, 800, 3), 'ref_image'))
            ret.append(tf.placeholder(tf.float32, (4,), 'ref_box'))
        if cfg.MODE_THIRD_STAGE:
            ret.append(tf.placeholder(tf.float32, (257, 7, 7), 'ff_gt_tracklet_feat'))
            ret.append(tf.placeholder(tf.float32, (None, 257, 7, 7), 'active_tracklets_feats'))
            ret.append(tf.placeholder(tf.float32, (None, 4), 'active_tracklets_boxes'))
            ret.append(tf.placeholder(tf.float32, (), 'tracklet_distance_threshold'))
        if cfg.EXTRACT_GT_FEATURES:
            ret.append(tf.placeholder(tf.float32, (None, 4,), 'roi_boxes'))
        return ret

    def backbone(self, image):
        c2345 = base_model.resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
        with base_model.backbone_scope(freeze=cfg.BACKBONE.FREEZE_AT > 3):
            p23456 = fpn_model.fpn_model('fpn', c2345)
        return p23456, c2345

    def rpn(self, image, features, inputs):
        if cfg.EXTRACT_GT_FEATURES:
            boxes = inputs['roi_boxes']
            return frcnn_model.BoxProposals(boxes), tf.constant(0, dtype=tf.float32)

        if cfg.BACKBONE.FREEZE_AT > 3:
            with freeze_variables(stop_gradient=False, skip_collection=True):
                return super().rpn(image, features, inputs)
        else:
            return super().rpn(image, features, inputs)

    def roi_heads(self, image, ref_image, ref_features, ref_box, features, proposals, targets,
                  precomputed_ref_features=None, precomputed_ref_mesh_features=None):
        image_shape2d = tf.shape(image)[2:]  # h,w
        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        if self.training:
            proposals = frcnn_model.sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)

        fastrcnn_head_func = getattr(frcnn_model, cfg.FPN.FRCNN_HEAD_FUNC)
        if precomputed_ref_features is None:
            roi_aligned_ref_features = fpn_model.multilevel_roi_align(ref_features[:4], ref_box[tf.newaxis], 7)
            ref_mesh_features = self.get_mesh_features(roi_aligned_ref_features)
            all_ref_features = tf.concat((roi_aligned_ref_features, ref_mesh_features), axis=1)
        else:
            roi_aligned_ref_features = precomputed_ref_features[tf.newaxis]
            ref_mesh_features = precomputed_ref_mesh_features[tf.newaxis]
            all_ref_features = tf.concat((roi_aligned_ref_features, ref_mesh_features), axis=1)

        if cfg.MODE_SHARED_CONV_REDUCE:
            scope = tf.get_variable_scope()
        else:
            scope = ""

        assert cfg.FPN.CASCADE

        def roi_func(boxes, already_aligned_features=None):
            if already_aligned_features is None:
                aligned_features = fpn_model.multilevel_roi_align(features[:4], boxes, 7)
                target_mesh_features = self.get_mesh_features(aligned_features)
                all_features = tf.concat((aligned_features, target_mesh_features), axis=1)
            else:
                # for hard example mining
                all_features = already_aligned_features
            # tf.shape(aligned_features) -> number of of box proposals
            tiled = tf.tile(all_ref_features, [tf.shape(all_features)[0], 1, 1, 1])
            # for each box the ref_features are concatenated with the aligned features
            concat_features = tf.concat((tiled, all_features), axis=1)

            with argscope(Conv2D, data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer(
                              scale=2.0, mode='fan_out',
                              distribution='untruncated_normal' if common_util.get_tf_version_tuple() >= (
                                      1, 12) else 'normal')):
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    reduced_features = Conv2D('conv_reduce', concat_features, 256, 1, activation=None)
            return reduced_features

        fastrcnn_head = cascade_model.CascadeRCNNHead(
            proposals, roi_func, fastrcnn_head_func,
            (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS)

        if cfg.EXTRACT_GT_FEATURES:
            # get boxes and features for each of the three cascade stages!
            b0 = proposals.boxes
            b1, b2, _ = fastrcnn_head._cascade_boxes
            f0 = fpn_model.multilevel_roi_align(features[:4], b0, 7)
            f1 = fpn_model.multilevel_roi_align(features[:4], b1, 7)
            f2 = fpn_model.multilevel_roi_align(features[:4], b2, 7)
            tf.concat([b0, b1, b2], axis=0, name="boxes_for_extraction")
            tf.concat([f0, f1, f2], axis=0, name="features_for_extraction")

        if self.training:
            all_losses = fastrcnn_head.losses()

            if cfg.MEASURE_IOU_DURING_TRAINING:
                decoded_boxes = fastrcnn_head.decoded_output_boxes()
                decoded_boxes = box_model.clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
                label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
                final_boxes, final_scores, final_labels = frcnn_model.fastrcnn_predictions(
                    decoded_boxes, label_scores, name_scope='output_train')
                # if predictions are empty, this might break...
                # to prevent, stack dummy box
                boxes_for_iou = tf.concat([final_boxes[:1], tf.constant([[0.0, 0.0, 1.0, 1.0]],
                                                                        dtype=tf.float32)], axis=0)
                from examples.FasterRCNN.utils.box_ops import pairwise_iou
                iou_at_1 = tf.identity(pairwise_iou(gt_boxes[:1], boxes_for_iou)[0, 0], name="train_iou_at_1")
                add_moving_summary(iou_at_1)

            return all_losses
        else:
            decoded_boxes = fastrcnn_head.decoded_output_boxes()
            decoded_boxes = box_model.clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
            label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
            final_boxes, final_scores, final_labels = frcnn_model.fastrcnn_predictions(
                decoded_boxes, label_scores, name_scope='output')
            return []

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        # oole: Transforms image, expands dim, transposes [0,3,1,2]
        #       also normalizes image with image mean and stdev
        image = self.preprocess(inputs['image'])  # 1CHW

        fpn_features, _ = self.backbone(image)

        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            ref_features = None
            ref_box = None
            ref_image = None
        else:
            # oole: the template image with the reference bounding box
            ref_image = self.preprocess(inputs['ref_image'])  # 1CHW
            ref_box = inputs['ref_box']
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                ref_features, _ = self.backbone(ref_image)

        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
        if cfg.EXTRACT_GT_FEATURES:
            anchor_inputs["roi_boxes"] = inputs["roi_boxes"]
        # Returns the box proposals and the box_proposal losses.
        proposals, rpn_losses = self.rpn(image, fpn_features, anchor_inputs)  # inputs?

        second_stage_features = fpn_features
        targets = [inputs[k] for k in ['gt_boxes', 'gt_labels'] if k in inputs]

        precomputed_ref_features = None
        precomputed_ref_mesh_features = None
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            precomputed_ref_features = inputs['ref_features']
            precomputed_ref_mesh_features = inputs['ref_mesh_features']

        # Extend proposals by previous frame detections
        if not self.training and cfg.MODE_THIRD_STAGE and cfg.EXTEND_PROPOSALS_BY_ACTIVE_TRACKLETS:
            proposal_boxes = proposals.boxes
            tracklet_boxes = inputs['active_tracklets_boxes']
            concat_boxes = tf.concat([proposal_boxes, tracklet_boxes], axis=0)
            proposals = frcnn_model.BoxProposals(concat_boxes)

        head_losses = self.roi_heads(image, ref_image, ref_features, ref_box, second_stage_features, proposals, targets,
                                     precomputed_ref_features=precomputed_ref_features,
                                     precomputed_ref_mesh_features=precomputed_ref_mesh_features)

        if cfg.MODE_THIRD_STAGE:
            self._run_third_stage(inputs, second_stage_features, tf.shape(image)[2:4])

        if self.training:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
            total_cost = tf.add_n(
                rpn_losses + head_losses + [wd_cost], 'total_cost')
            add_moving_summary(total_cost, wd_cost)
            return total_cost

    def _run_third_stage(self, inputs, second_stage_features, image_hw):
        boxes, scores = get_tensors_by_names(['output/boxes', 'output/scores'])
        # let's fix (as in finalize) the boxes, so we can roi align only one time
        aligned_features_curr = fpn_model.multilevel_roi_align(second_stage_features[:4], boxes, 7)

        mesh_features_curr = self.get_mesh_features(aligned_features_curr)
        combined_features = tf.concat((aligned_features_curr, mesh_features_curr), axis=1)
        # these also need to be extracted!
        combined_features = tf.identity(combined_features, name='third_stage_features_out')

        ff_gt_tracklet_scores, _ = self._score_for_third_stage(ref_feats=inputs['ff_gt_tracklet_feat'][tf.newaxis],
                                                               det_feats=combined_features)
        tf.identity(ff_gt_tracklet_scores, name='ff_gt_tracklet_scores')
        sparse_tracklet_scores, tracklet_score_indices = self._score_for_third_stage(
            ref_feats=inputs['active_tracklets_feats'], det_feats=combined_features,
            dense=False, ref_boxes=inputs['active_tracklets_boxes'], det_boxes=boxes, image_hw=image_hw,
            tracklet_distance_threshold=inputs['tracklet_distance_threshold'])
        tf.identity(sparse_tracklet_scores, name='sparse_tracklet_scores')
        tf.identity(tracklet_score_indices, name='tracklet_score_indices')

    def _score_for_third_stage(self, ref_feats, det_feats, dense=True, ref_boxes=None, det_boxes=None, image_hw=None,
                               tracklet_distance_threshold=0.08):
        # build all pairs
        n_refs = tf.shape(ref_feats)[0]
        n_dets = tf.shape(det_feats)[0]

        active_tracklets_tiled = tf.tile(ref_feats[:, tf.newaxis], multiples=[1, n_dets, 1, 1, 1])
        dets_tiled = tf.tile(det_feats[tf.newaxis], multiples=[n_refs, 1, 1, 1, 1])
        concated = tf.concat([active_tracklets_tiled, dets_tiled], axis=2)

        if not dense:
            # use boxes to prune the connectivity
            assert ref_boxes is not None
            assert det_boxes is not None
            assert image_hw is not None

            def xyxy_to_cxcywh(boxes_xyxy):
                wh = boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]
                c = boxes_xyxy[:, :2] + wh / 2
                boxes_cwh = tf.concat((c, wh), axis=1)
                return boxes_cwh

            active_tracklets_boxes_cxcywh = xyxy_to_cxcywh(ref_boxes)
            boxes_cxcywh = xyxy_to_cxcywh(det_boxes)

            # normalize by image size
            h = image_hw[0]
            w = image_hw[1]
            norm = tf.cast(tf.stack([w, h, w, h], axis=0), tf.float32)
            diffs = tf.abs(active_tracklets_boxes_cxcywh[:, tf.newaxis] - boxes_cxcywh[tf.newaxis]) / norm[
                tf.newaxis, tf.newaxis]

            # use distances of boxes, first frame scores ("scores") to prune
            thresholds = tf.stack([tracklet_distance_threshold] * 4, axis=0)
            keep_mask = tf.reduce_all(diffs < thresholds, axis=2)

            indices = tf.where(keep_mask)
            flattened = tf.boolean_mask(concated, keep_mask)
        else:
            indices = None
            flattened = tf.reshape(
                concated, [tf.shape(concated)[0] * tf.shape(concated)[1]] + [int(x) for x in concated.shape[2:]])

        fastrcnn_head_func = getattr(frcnn_model, cfg.FPN.FRCNN_HEAD_FUNC)
        if cfg.MODE_SHARED_CONV_REDUCE:
            scope = tf.get_variable_scope()
        else:
            scope = ""
        all_posteriors = []
        # do this for every cascade stage
        for idx in range(3):
            with tf.variable_scope('cascade_rcnn_stage{}'.format(idx + 1), reuse=True):
                with argscope(Conv2D, data_format='channels_first'):
                    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                        reduced_features = Conv2D('conv_reduce', flattened, 256, 1, activation=None)
                    head_feats = fastrcnn_head_func('head', reduced_features)
                    with tf.variable_scope('outputs_new', reuse=True):
                        classification = FullyConnected('class', head_feats, 2)
                        posteriors = tf.nn.softmax(classification)
                        all_posteriors.append(posteriors)
        posteriors = (all_posteriors[0] + all_posteriors[1] + all_posteriors[2]) / tf.constant(3.0, dtype=tf.float32)
        scores = posteriors[:, 1]
        return scores, indices

    def get_inference_tensor_names(self):
        inp, out = super().get_inference_tensor_names()
        if cfg.USE_PRECOMPUTED_REF_FEATURES:
            inp.append('ref_features')
            inp.append('ref_mesh_features')
        else:
            inp.append('ref_image')
            inp.append('ref_box')
        if cfg.MODE_THIRD_STAGE:
            inp.append('ff_gt_tracklet_feat')
            inp.append('active_tracklets_feats')
            inp.append('active_tracklets_boxes')
            inp.append('tracklet_distance_threshold')
        return inp, out
