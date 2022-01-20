calculate_2_error() {
echo ---------- Calculating error for $5 ----------
echo Calculating errors for Model $1 and $2
python errors_2_models.py --first-prediction $1 --second-prediction $2 --first-plot-name "$3" --second-plot-name "$4" --error-name $5 --error-dir $6 --x-axis-size $7
echo Finished $5
echo ----------------------------------------------
}

calculate_4_error() {
echo ---------- Calculating error for $9 ----------
echo Calculating errors for Model $1, $2, $3, $4
python errors_4_models.py --first-prediction $1 --second-prediction $2 --third-prediction $3  --fourth-prediction $4 --first-plot-name "$5" --second-plot-name "$6" --third-plot-name "$7" --fourth-plot-name "$8" --error-name $9 --error-dir $10 --x-axis-size $11
echo Finished $9
echo ----------------------------------------------
}


# Execute calculate_predictions.sh first.
# (split into two scripts, so that predictions and errors can be performed in different environments)
# Calculates the error for the given model predictions nd the specified data
# Extrapolation:
calculate_2_error results_fpn2mesh/cheb_edge_result.npy results_fpn2mesh/cheb_l1_result.npy "FPN2Mesh Cheb Edge" "FPN2Mesh Cheb L1"         cheb_edge-vs-l1     errors_fpn2mesh 35
calculate_2_error results_fpn2mesh/spiral_edge_result.npy results_fpn2mesh/spiral_l1_result.npy "FPN2Mesh Spiral Edge" "FPN2Mesh Spiral L1" spiral_edge-vs-l1   errors_fpn2mesh 35
calculate_2_error results_fpn2mesh/spiral_edge_result.npy results_fpn2mesh/cheb_edge_result.npy "FPN2Mesh Spiral Edge" "FPN2Mesh Cheb Edge" spiral-vs-cheb_edge errors_fpn2mesh 35
calculate_2_error results_fpn2mesh/spiral_l1_result.npy results_fpn2mesh/cheb_l1_result.npy "FPN2Mesh Spiral L1" "FPN2Mesh Cheb L1"         spiral-vs-cheb_l1   errors_fpn2mesh 35

calculate_4_error results_fpn2mesh/spiral_edge_result.npy results_fpn2mesh/spiral_l1_result.npy results_fpn2mesh/cheb_edge_result.npy results_fpn2mesh/cheb_l1_result.npy "FPN2Mesh Spiral Edge" "FPN2Mesh Spiral L1" "FPN2Mesh Cheb Edge" "FPN2Mesh Cheb L1" all errors_fpn2mesh 35

