run_argmax_tracker() {
echo ---------- Running ArgmaxTracker for $2, model: $1 ----------
python do_siam_mesh_rcnn_tracker.py --tracker ArgmaxTracker --model $1 --name $2
echo Finished ArgMaxtracker for $2, model: $1
echo ----------------------------------------------
}

run_threestage_tracker() {
echo ---------- Running ThreeStageTracker for $2, model: $1 ----------
python do_siam_mesh_rcnn_tracker.py --tracker ThreeStageTracker --model $1 --name $2
echo Finished ThreeStageTracker for $2, model: $1
echo ----------------------------------------------
}


run_argmax_tracker "/path/to.../" "spiral-edge"
run_threestage_tracker "/path.../to" "spiral-edge"

run_argmax_tracker "/path/to.../" "cheb-edge"
run_threestage_tracker "/path.../to" "cheb-edge"


