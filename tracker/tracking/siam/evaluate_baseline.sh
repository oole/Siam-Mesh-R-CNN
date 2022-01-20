run_argmax_tracker() {
echo ---------- Running ArgmaxTracker for $1 ----------
python do_tracker.py --tracker ArgmaxTracker --model $1
echo Finished ArgMaxtracker for $1
echo ----------------------------------------------
}

run_threestage_tracker() {
echo ---------- Running ThreeStageTracker for $1 ----------
python do_tracker.py --tracker ThreeStageTracker --model $1
echo Finished ThreeStageTracker for $1
echo ----------------------------------------------
}


run_argmax_tracker "/home/oole/conv-track/final_model_dir/baseline/model-250000"
run_threestage_tracker "/home/oole/conv-track/final_model_dir/baseline/model-250000"
