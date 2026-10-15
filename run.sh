#!/usr/bin/env bash
# Runs scripts/demo_video.py in the conda env, on the GPU with the most free memory. Run it from
# the repository root; every option but --env and --gpu goes to demo_video.py:
#
#   bash run.sh --input_video dance.mp4 --output_dir results
set -e

ENV=worldhmr-fastcam
GPU=
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --env) ENV=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} -gt 0 ] || { sed -n '2,5p' "$0"; exit 1; }

eval "$(conda shell.bash hook)"
conda activate "$ENV"

if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fi
echo "GPU $GPU, conda env $ENV"
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/demo_video.py "${ARGS[@]}"
