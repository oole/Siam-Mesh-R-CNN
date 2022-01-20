calculate_predictions() {
echo ---------- Calculating error for $1 ----------
echo Running FPN2Mesh
python evaluate_fpn2mesh.py --load $1 --eval-name $2 --conv-mode $3
echo Predictions for $1 on $2 saved
echo ----------------------------------------------
}

# Calculates the prediction using the specified model (-checkpoint) and the specified data
# CoMA template
#calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_norm/dicts/tp_coma_fpn2mesh_norm_cheb_edge_220210-0155-310epochs.npz       cheb_edge   cheb
#calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_norm/dicts/tp_coma_fpn2mesh_norm_cheb_l1-only_220210-0155-310epochs.npz    cheb_l1     cheb
#calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_norm/dicts/tp_coma_fpn2mesh_norm_spiral_edge_220210-0155-310epochs.npz     spiral_edge spiral
#calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_norm/dicts/tp_coma_fpn2mesh_norm_spiral_l1-only_220210-0155-310epochs.npz  spiral_l1   spiral

# FLAME template
calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_flame/dicts/tp_coma_fpn2mesh_norm_cheb_edge_FLAME_220210-1845-310epochs.npz        cheb_edge   cheb
calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_flame/dicts/tp_coma_fpn2mesh_norm_cheb_l1-only_FLAME_220210-1845-310epochs.npz     cheb_l1     cheb
calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_flame/dicts/tp_coma_fpn2mesh_norm_spiral_edge_FLAME_220210-1845-310epochs.npz      spiral_edge spiral
calculate_predictions /home/oole/conv-track/final_model_dir/fpn2mesh_flame/dicts/tp_coma_fpn2mesh_norm_spiral_l1-only_FLAME_220210-1845-310epochs.npz   spiral_l1   spiral