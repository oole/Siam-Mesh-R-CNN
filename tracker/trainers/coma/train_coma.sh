train_coma() {
echo ---------- Calculating error for $1 ----------
echo Training CoMA for $1, with $2, $3
python train_coma.py --logdir $1 --loss $2 --conv-mode $3
echo Training for CoMA for $1 finished.
echo ----------------------------------------------
}

train_coma "../../computed/coma_norm_cheb_edge"   edge    cheb
train_coma "../../computed/coma_norm_cheb_l1"     l1-only cheb
train_coma "../../computed/coma_norm_spiral_edge" edge    spiral
train_coma "../../computed/coma_norm_spiral_l1"   l1-only spiral