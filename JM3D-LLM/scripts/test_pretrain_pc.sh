#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat
MODEL_VERSION=vicuna-7b

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########
# CUDA_VISIBLE_DEVICES=3 \
# python llava/eval/run_llava_pc.py \
#     --model_path ./checkpoints/llava-lightning-7b-objaverse-pretrain-no3Dword-nofreeze_vis_backbone/checkpoint-2400 \
#     --pc_file ./data/Objaverse/Cap3D_pcs_pt/252f3b3f5cd64698826fc1ab42614677.pt \
#     --pc_dataset objaverse \
#     --query "Relay a brief, clear account of the point cloud shown."


# MODEL_PATH='./checkpoints/llava-lightning-7b-objaverse-pretrain-no3Dword-nofreeze_vis_backbone'
MODEL_PATH='./checkpoints/stage1_ckpt'
PC_FILE='/home/myw/wuchangli/yk/JM3D/JM3D-LLM/data/objaverse_data/4887da0aab51406dab3c5cb69ec82404_8192.npy'
CUDA_VISIBLE_DEVICES=3 \
python llava/eval/run_llava_pc.py \
    --model_path $MODEL_PATH \
    --pc_file $PC_FILE \
    --pc_dataset objaverse \
    --query "What is it?" \
    --torch_dtype float32

# CUDA_VISIBLE_DEVICES=3 \
# python llava/eval/run_llava_pc.py \
#     --model_path ./checkpoints/llava-lightning-7b-objaverse-pretrain-nofreeze_vis_backbone \
#     --pc_file data/ModelNet \
#     --pc_dataset modelnet \
#     --query "Can you describe this point cloud?"

# CUDA_VISIBLE_DEVICES=3 \
# python llava/eval/run_llava_pc.py \
#     --model_path ./checkpoints/vicuna-7b-pretrain/checkpoint-400 \
#     --pc_file data/ScanNet/scans/scene0000_00/scene0000_00_vh_clean_2.ply \
#     --pc_dataset scannet \
#     --query "Can you describe this point cloud? Is it a person or some object?"