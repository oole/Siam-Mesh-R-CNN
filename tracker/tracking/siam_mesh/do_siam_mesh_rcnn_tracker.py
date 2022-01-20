import argparse
import os

from got10k.experiments.custom import ExperimentCustom

from tracker.tracking.got10 import experiment as exp
from tracker.tracking.siam_mesh.siam_mesh_rcnn_argmax_tracker import ArgmaxConvTracker
from tracker.tracking.siam_mesh.siam_mesh_rcnn_three_stage_tracker import ThreeStageConvTracker

RESULT_DIR = 'results/'
REPORT_DIR = 'reports/'

parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int, help='first video index to process', default=0)
parser.add_argument('--end_idx', type=int, help='last video index to process (exclusive)', default=None)

# TDPA parameters. You can just leave them at the default values which will work well on a wide range of datasets
parser.add_argument('--tracklet_distance_threshold', type=float, default=0.06)
parser.add_argument('--tracklet_merging_threshold', type=float, default=0.3)
parser.add_argument('--tracklet_merging_second_best_relative_threshold', type=float, default=0.3)
parser.add_argument('--ff_gt_score_weight', type=float, default=0.1)
parser.add_argument('--ff_gt_tracklet_score_weight', type=float, default=0.9)
parser.add_argument('--location_score_weight', type=float, default=7.0)

parser.add_argument('--model', type=str, default="oole-resnet50-fpn2mesh",
                    help='one of "best", "nohardexamples", or "gotonly"')
parser.add_argument('--tracker', type=str, default='ThreeStageTracker',
                    help="One of ArgmaxTracker or ThreeStageTracker")
parser.add_argument('--n_proposals', type=int, default=None)
parser.add_argument('--resolution', type=str, default=None)
parser.add_argument('--visualize_tracker', action='store_true',
                    help='use visualization of tracker (recommended over --visualize_experiment)')
parser.add_argument('--visualize_experiment', action='store_true',
                    help='use visualization of got experiment (not recommended, usually --visualize_tracker is better)')
parser.add_argument('--custom_dataset_name', type=str, default=None)
parser.add_argument('--custom_dataset_root_dir', type=str, default=None)
parser.add_argument('--main', type=str)
parser.add_argument('--computed-dir', default="../../computed/tracker")
parser.add_argument('--name', type=str, help="The name of the run")
parser.add_argument('--conv-mode', default="spiral")
args = parser.parse_args()


def build_tracker():
    if args.tracker == "ArgmaxTracker":
        return ArgmaxConvTracker(model=args.model, conv_mode=args.conv_mode)
    elif args.tracker == "ThreeStageTracker":
        pass
    else:
        assert False, ("Unknown tracker", args.tracker)

    tracklet_param_str = str(args.tracklet_distance_threshold) + "_" + str(args.tracklet_merging_threshold) + "_" + \
                         str(args.tracklet_merging_second_best_relative_threshold)
    if args.n_proposals is not None:
        tracklet_param_str += "_proposals" + str(args.n_proposals)
    if args.resolution is not None:
        tracklet_param_str += "_resolution-" + str(args.resolution)
    if args.model != "best":
        tracklet_param_str = os.path.basename(args.model) + "_" + tracklet_param_str
    if args.visualize_tracker:
        tracklet_param_str2 = "viz_" + tracklet_param_str
    else:
        tracklet_param_str2 = tracklet_param_str
    param_str = tracklet_param_str2 + "_" + str(args.ff_gt_score_weight) + "_" + \
                str(args.ff_gt_tracklet_score_weight) + "_" + str(args.location_score_weight)

    name = "ThreeStageTracker_" + param_str
    tracker = ThreeStageConvTracker(tracklet_distance_threshold=args.tracklet_distance_threshold,
                                    tracklet_merging_threshold=args.tracklet_merging_threshold,
                                    tracklet_merging_second_best_relative_threshold=
                                    args.tracklet_merging_second_best_relative_threshold,
                                    ff_gt_score_weight=args.ff_gt_score_weight,
                                    ff_gt_tracklet_score_weight=args.ff_gt_tracklet_score_weight,
                                    location_score_weight=args.location_score_weight,
                                    name=name,
                                    do_viz=args.visualize_tracker,
                                    model=args.model,
                                    n_proposals=args.n_proposals,
                                    resolution=args.resolution,
                                    conv_mode=args.conv_mode)
    return tracker


def main_custom():
    custom_dataset_root_dir = args.custom_dataset_root_dir
    assert custom_dataset_root_dir is not None
    custom_dataset_name = args.custom_dataset_name
    assert custom_dataset_name is not None
    tracker = build_tracker()
    experiment = ExperimentCustom(
        root_dir=custom_dataset_root_dir,
        name=custom_dataset_name
    )
    experiment.run(tracker, visualize=args.visualize_experiment)


def main_chokepoint():
    result_dir_path = os.path.join(args.computed_dir, RESULT_DIR)
    report_dir_path = os.path.join(args.computed_dir, REPORT_DIR)
    tracker = build_tracker()
    experiment = exp.ExperimentChokepoint(
        name="siam_mesh_rcnn-" + args.name,
        result_dir=result_dir_path,
        report_dir=report_dir_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    experiment.run(tracker, visualize=args.visualize_experiment)
    experiment.report([tracker.name])


if __name__ == "__main__":
    # assert args.main is not None, "--main not supplied, e.g. --main main_otb"
    # TODO: Use pre-trained backbone from SiamRCNN, restoring only these weights
    # TODO: Concatenate mesh feature vector to ROIAlignedFeatures.
    main_chokepoint()

    eval(args.main + "()")
