export MODEL_PATH=deepseek-ai/DeepSeek-V2-Lite
DATASET=gqa NUM_SAMPLES=1024 TOKEN_PER_SAMPLE=2048 bash scripts/run_collect_scores.sh
DATASET=coco NUM_SAMPLES=1024 TOKEN_PER_SAMPLE=2048 bash scripts/run_collect_scores.sh
DATASET=m4 NUM_SAMPLES=1024 TOKEN_PER_SAMPLE=2048 bash scripts/run_collect_scores.sh
DATASET=video_mmmu NUM_SAMPLES=1024 TOKEN_PER_SAMPLE=2048 bash scripts/run_collect_scores.sh

