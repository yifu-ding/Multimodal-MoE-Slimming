
SCORES_PATH=/home/dyf/code/distill/MAES/storage/scores/qwen3-vl-30b-a3b_video_mmmu-num_128-token_2048-rel_l2-0506-135316/scores.pt MODEL_NAME=qwen3-vl-30b-a3b-instruct SUFFIX=mmmu-num_128-token_2048-rel_l2-0506-135316 bash scripts/sweep_video_prune_eval.sh

SCORES_PATH=/home/dyf/code/distill/MAES/storage/scores/qwen3-vl-30b-a3b_video_mmmu-num_128-token_2048-rel_l2-0506-135316/scores.pt MODEL_NAME=qwen3-vl-30b-a3b-instruct SUFFIX=mmmu-num_128-token_2048-rel_l2-0506-135316-p0.3 PRUNE_RATIO=0.3 bash scripts/sweep_video_prune_eval.sh

SCORES_PATH=/home/dyf/code/distill/MAES/storage/scores/internvl3_5-30b-a3b_video_mmmu-num_128-token_2048-rel_l2-0506-140130/scores.pt MODEL_NAME=internvl3_5-30b-a3b-hf NUM_SAMPLES=1000 SWEEP_INTER_METHODS=loss_smooth_2 LAYERWISE_LOSS_KEY=layerwise_second_order_sum SUFFIX=mmmu-num_128-token_2048-rel_l2-0506-140130-p0.5 bash scripts/sweep_video_prune_eval.sh

SCORES_PATH=/home/dyf/code/distill/MAES/storage/scores/internvl3_5-30b-a3b_video_mmmu-num_128-token_2048-rel_l2-0506-140130/scores.pt MODEL_NAME=internvl3_5-30b-a3b-hf NUM_SAMPLES=1000 SWEEP_INTER_METHODS=loss_smooth_2 LAYERWISE_LOSS_KEY=layerwise_second_order_sum SUFFIX=mmmu-num_128-token_2048-rel_l2-0506-140130-p0.3  PRUNE_RATIO=0.3 bash scripts/sweep_video_prune_eval.sh
