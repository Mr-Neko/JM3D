master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
# point_backbone_ckpt=/data/haowei/pointbert-nosmo.pt
# point_backbone_ckpt=/home/myw/haowei/v1.1_pointbert.pt
point_backbone_ckpt=/home/myw/haowei/v1.1_pointbert_replace.pt
# point_backbone_ckpt=/home/myw/wuchangli/PointLLM/checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.1.pt
# point_backbone_ckpt=/home/myw/wuchangli/PointLLM/checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.1_noisy_1_10.pt

# dir_path=PointLLM
model_name_or_path=checkpoints/stage1_ckpt
data_path=data/objaverse_data
# anno_path=data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
anno_path_stage2=data/anno_data/PointLLM_complex_instruction_70K.json
output_dir_stage2=checkpoints/stage2_ckpt
# point_backbone_ckpt=$model_name_or_path/point_bert_v1.2.pt
# point_backbone_ckpt=/data/haowei/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt

# cd $dir_path


# PYTHONPATH=$dir_path:$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 \
torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port \
    llava/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path_stage2 \
    --output_dir $output_dir_stage2 \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --training_stage 2 \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --use_color True