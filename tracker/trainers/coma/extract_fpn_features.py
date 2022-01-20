from tensorpack import PredictConfig, get_model_loader, OfflinePredictor

from tracker.config import config as cfg, finalize_configs
from tracker.data.data import DetectionDataset
from tracker.model.siam.track_model import ResNetFPNModel
from PIL import Image
import numpy as np
import glob
import os
import xmltodict
from tqdm import tqdm


if __name__ == "__main__" :
    load = ("/mnt/storage/Msc/conv-track/pre-trained/COCO-MaskRCNN-R50FPN4xGNCasAug.npz")

    # init tensorpack model
    # cfg.freeze(False)
    DetectionDataset()  # initialize the config with information from our dataset
    cfg.EXTRACT_GT_FEATURES = True
    cfg.MODE_TRACK = False

    extract_model = ResNetFPNModel()
    extract_ff_feats_cfg = PredictConfig(
        model=extract_model,
        session_init=get_model_loader(load),
        input_names=['image', 'roi_boxes'],
        output_names=['rpn/feature'])
    finalize_configs(is_training=False)
    feature_extractor_function = OfflinePredictor(extract_ff_feats_cfg)

    cfg.EXTRACT_GT_FEATURES = False
    cfg.MODE_TRACK = True
    cfg.USE_PRECOMPUTED_REF_FEATURES = True
    # 1. Iterate over all training images, with bbox, and extract the aligned fpn feautres
    # 2. Store these in same format as sequences
    # 3. Train mesh reconstruction network with the fpn features as input and the mesh features as goal
    # 4. The fpn features represent the output of the encoder - which is actually the resnet/fpn network
    # 5. The decoder obtained this way should be usable as mesh decoder directly on the roi_aligned features!!
    # 6. profit.
    # Features for eax box are [1,256,7,7]
    # convolve and reduce, flatten and use as input for decoder.

    subset = "val"
    subset_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation", "G1", subset)
    paths = sorted(glob.glob(subset_path + "*/*/*/*"))
    vid_names_1 = [path for path in paths if "xml" not in path and "seq" in path]
    vid_names_1 = ["/".join(v.split("/")[-3:]) for v in vid_names_1]
    paths = sorted(glob.glob(subset_path + "*/*/*"))
    vid_names_2 = [path for path in paths if "xml" not in path and "seq" in path]
    vid_names_2 = ["/".join(v.split("/")[-2:]) for v in vid_names_2]
    vid_names = list(vid_names_2 + vid_names_1)

    for vid_name in tqdm(vid_names):
        ann_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation/G1/", subset, vid_name)
        ann_files = sorted(glob.glob(ann_path + "/*.xml"))
        # randomly select two files
        for ann_file in ann_files:
            ann = xmltodict.parse(open(ann_file).read())["annotation"]
            def obj_data_to_bbox(ann):
                obj_ann = ann['object']
                bbox = obj_ann['bndbox']
                x1 = bbox['xmin']
                y1 = bbox['ymin']
                x2 = bbox['xmax']
                y2 = bbox['ymax']
                box = [x1, y1, x2, y2]
                return box


            def obj_file_path(ann):
                sub_path = ann['folder']
                fname = ann['filename']
                full_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, sub_path, fname + ".jpg")
                return full_path

            def get_feat_path(ann_file: str):
                bb_feat_path = ann_file.replace("annotation", "bbox_features")
                bb_feat_path = bb_feat_path.replace(".xml", ".npy")
                return bb_feat_path

            bbox = np.asarray(obj_data_to_bbox(ann))
            img_path = obj_file_path(ann)

            image = Image.open(img_path)
            img = np.array(image)[..., ::-1]
            feats = feature_extractor_function(img, bbox[np.newaxis])
            feat_path = get_feat_path(ann_file)
            if not os.path.isdir(os.path.dirname(feat_path)):
                os.makedirs(os.path.dirname(feat_path))

            np.save(get_feat_path(ann_file), feats)




