calculate_predictions() {
echo ---------- Calculating error for $1 ----------
echo Running FPN2Mesh
python evaluate_coma.py --load $1 --eval-name $2 --conv-mode $3
echo Predictions for $1 on $2 saved
echo ----------------------------------------------
}

# Calculates the prediction using the specified model (-checkpoint) and the specified data
# CoMA template
#calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_norm/dicts/tp_coma_lat-49_norm_cheb_edge-loss-2202101358-300epochs.npz     cheb_edge   cheb
#calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_norm/dicts/tp_coma_lat-49_norm_cheb_l1-loss-2202101358-300epochs.npz       cheb_l1     cheb
#calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_norm/dicts/tp_coma_lat-49_norm_spiral_edge-loss-2202101358-300epochs.npz   spiral_edge spiral
#calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_norm/dicts/tp_coma_lat-49_norm_spiral_l1-loss-2202101358-300epochs.npz     spiral_l1   spiral

# FLAME template:
calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_flame/dicts/tp_coma_lat-49_norm_cheb_edge-loss_FLAME_220210-2015-300epochs.npz     cheb_edge   cheb
calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_flame/dicts/tp_coma_lat-49_norm_cheb_l1-loss_FLAME_220210-2015-300epochs.npz       cheb_l1     cheb
calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_flame/dicts/tp_coma_lat-49_norm_spiral_edge-loss_FLAME_220210-2015-300epochs.npz   spiral_edge spiral
calculate_predictions /home/oole/conv-track/final_model_dir/coma-autoencoder-lat49_flame/dicts/tp_coma_lat-49_norm_spiral_l1-loss_FLAME_220210-2015-300epochs.npz     spiral_l1   spiral