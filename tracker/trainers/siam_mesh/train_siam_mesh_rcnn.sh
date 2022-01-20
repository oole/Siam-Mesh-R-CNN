train_siam_rcnn() {
echo ---------- Calculating error for $1 ----------
echo Training Siam Mesh R-CNN for $1, with $3,$4
python train_siam_mesh_rcnn.py ---logdir $1 --load "$2" --fpn-conv-mode $3 --coma-conv-mode $4
echo Training for Siam Mesh R-CNN for $1 finished.
echo ----------------------------------------------
}

train_siam_rcnn "../../computed/siam_mesh_rcnn_all-spiral"  "../../pre-trained/COCO-MaskRCNN-R50FPN4xGNCasAug.npz ../../computed/fpn2mesh_norm_spiral_edge-310epochs.npz ../../computed/coma_spiral_edge-loss-300epochs.npz" spiral spiral
train_siam_rcnn "../../computed/siam_mesh_rcnn_all-spiral"  "../../pre-trained/COCO-MaskRCNN-R50FPN4xGNCasAug.npz ../../computed/fpn2mesh_norm_cheb_edge-310epochs.npz ../../computed/coma_cheb_edge-loss-300epochs.npz" cheb cheb