#!/bin/bash
# Pretraining

GPUs=1
torchrun --nnodes=1 --nproc_per_node=$GPUs --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./backbones/vicuna-7b \
    --version v1 \
    --data_path data/objaverse_data \
    --anno_path data/anno_data/PointLLM_brief_description_660K_filtered.json \
    --vision_tower /home/myw/haowei/v1.1_pointbert_replace.pt \
    --bf16 True \
    --output_dir ./checkpoints/stage1_ckpt \
    --num_train_epochs 3 \
    --num_gpus $GPUs \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb