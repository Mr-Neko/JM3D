#!/bin/bash
# Pretraining

GPUs=3
torchrun --nnodes=1 --nproc_per_node=$GPUs --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path ./backbones/vicuna-7b \
    --version v1 \
    --data_path ./data/Objaverse/pc_chat_Cap3D_660k.json \
    --pc_folder ./data/Objaverse/Cap3D_pcs_pt \
    --vision_tower ./backbones/pointmlp/pointmlp_backbone.pt \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./checkpoints/llava-lightning-7b-objaverse-pretrain-no3Dword-nofreeze_vis_backbone \
    --num_train_epochs 1 \
    --num_gpus $GPUs \
    --per_device_train_batch_size 22 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb