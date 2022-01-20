train_fpn2mesh() {
echo ---------- Calculating error for $1 ----------
echo Training FPN2Mesh for $1, with $2, $3
python train_fpn2mesh.py --logdir $1 --loss $2 --conv-mode $3
echo Training for FPN2Mesh for $1 finished.
echo ----------------------------------------------
}

train_fpn2mesh "../../computed/fpn2mesh_norm_cheb_edge"   edge    cheb
train_fpn2mesh "../../computed/fpn2mesh_norm_cheb_l1"     l1-only cheb
train_fpn2mesh "../../computed/fpn2mesh_norm_spiral_edge" edge    spiral
train_fpn2mesh "../../computed/fpn2mesh_norm_spiral_l1"   l1-only spiral