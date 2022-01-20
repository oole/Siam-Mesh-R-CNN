convert_to_dict() {
echo ---------- Calculating error for $1 ----------
echo Training FPN2Mesh for $1, with $2, $3
python convert_checkpoint_to_dict.py --checkpoint-dir $1 --checkpoint
echo Training for FPN2Mesh for $1 finished.
echo ----------------------------------------------
}

convert_to_dict "../../computed/coma_norm_cheb_edge"
convert_to_dict "../../computed/coma_norm_cheb_l1"
convert_to_dict "../../computed/coma_norm_spiral_edge"
convert_to_dict "../../computed/coma_norm_spiral_l1"

convert_to_dict "../../computed/fpn2mesh_norm_cheb_edge"
convert_to_dict "../../computed/fpn2mesh_norm_cheb_l1"
convert_to_dict "../../computed/fpn2mesh_norm_spiral_edge"
convert_to_dict "../../computed/fpn2mesh_norm_spiral_l1"