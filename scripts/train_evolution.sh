#!/bin/bash
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN

python -m evolutionary_rlhf.evolution_trainer \
    --evolution_config configs/evolution_config.yaml \
    --ppo_config configs/ppo_config.yaml \
    --model_config configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config configs/task_config.yaml \
    --num_gpus 8 \
    --num_nodes 1
