#!/bin/bash


accelerate launch --multi_gpu --num_processes=2 basic_training.py \
        --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
        --model_type qwen2_moe \
        --dataset_name vinhpx/function-calling-mixed \
        --output_dir ./output-distributed \
        --precision fp8 \
        --batch_size 1 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 3 \
        --sub_dataset_percent 0.3 \
        --learning_rate 2e-5 \
        --with_tracking \
        --wandb_project qwen-finetune \
        --wandb_name qwen_distributed_run \
        --distributed