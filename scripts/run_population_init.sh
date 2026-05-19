#!/bin/bash
export RAY_DEDUP_LOGS=0

python -m evolutionary_rlhf.evolution_trainer \
    --evolution_config configs/evolution_config.yaml \
    --ppo_config configs/ppo_config.yaml \
    --model_config configs/model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config configs/task_config.yaml \
    --init_only
