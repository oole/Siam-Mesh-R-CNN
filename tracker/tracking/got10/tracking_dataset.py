from __future__ import absolute_import, print_function, unicode_literals

import glob
import os

import numpy as np
import six
import xmltodict

from tracker.config import config as cfg


class Chokepoint(object):
    def __init__(self, subset="test"):
        super(Chokepoint, self).__init__()
        subset_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation", "G1", subset)
        paths = sorted(glob.glob(subset_path + "*/*/*/*"))
        sequences_1 = [path for path in paths if "xml" not in path and "seq" in path]
        sequences_1 = ["/".join(v.split("/")[-3:]) for v in sequences_1]
        paths = sorted(glob.glob(subset_path + "*/*/*"))
        sequences_2 = [path for path in paths if "xml" not in path and "seq" in path]
        sequences_2 = ["/".join(v.split("/")[-2:]) for v in sequences_2]
        sequences = list(sequences_2 + sequences_1)

        self.seq_dirs = [os.path.join(cfg.DATA.CHOKEPOINT_ROOT, "annotation", "G1", subset, seq) for seq in sequences]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        seq_dir = self.seq_dirs[index]
        sequence_annotation_paths = sorted(glob.glob(seq_dir + "/*.xml"))

        # sequence_annotation_paths = [os.path.join(seq_dir, fname) for fname in sequence_annotation_fnames]

        def obj_file_path(annotation):
            sub_path = annotation['folder']
            fname = annotation['filename']
            full_path = os.path.join(cfg.DATA.CHOKEPOINT_ROOT, sub_path, fname + ".jpg")
            return full_path

        def obj_data_to_bbox(annotation):
            object_ann = annotation['object']
            bbox = object_ann['bndbox']
            x1 = int(bbox['xmin'])
            y1 = int(bbox['ymin'])
            x2 = int(bbox['xmax'])
            y2 = int(bbox['ymax'])

            # OTB format is given as x,y,w,h, so we convert it
            box = [x1, y1, x2-x1, y2-y1]
            return box

        img_files = []
        annotations = []
        for ann_path in sequence_annotation_paths:
            ann = xmltodict.parse(open(ann_path).read())['annotation']
            img_files.append(obj_file_path(ann))
            annotations.append(obj_data_to_bbox(ann))

        annotations = np.asarray(annotations)
        assert len(img_files) == len(annotations)
        assert annotations.shape[1] == 4

        return img_files, annotations

    def __len__(self):
        return len(self.seq_names)
