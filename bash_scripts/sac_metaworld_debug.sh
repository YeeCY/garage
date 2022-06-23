#!/bin/bash

SCRIPT_DIR=$(dirname "$BASH_SOURCE")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")

ENV_NAME=$1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR

declare -a seeds=(0)

for seed in "${seeds[@]}"; do
  rm $CONDA_PREFIX/lib/python*/site-packages/mujoco_py/generated/mujocopy-buildlock
  mkdir -p ~/offline_c_learning/garage_metaworld_logs_debug/sac-"$ENV_NAME"/"$seed"
  export CUDA_VISIBLE_DEVICES=0
  python $PROJECT_DIR/metaworld_examples/sac_metaworld.py \
    --env_name "$ENV_NAME" \
    --seed "$seed" \
    --base_log_dir ~/offline_c_learning/garage_metaworld_logs_debug
done
