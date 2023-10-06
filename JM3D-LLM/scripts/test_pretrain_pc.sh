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

CUDA_VISIBLE_DEVICES=3 \
python llava/eval/run_llava_pc.py \
    --model_path ./checkpoints/llava-lightning-7b-objaverse-pretrain-no3Dword-nofreeze_vis_backbone \
    --pc_file ./data/Objaverse/Cap3D_pcs_pt/21cf8adec67b4e2b85f579e3cd3dcbc0.pt \
    --pc_dataset objaverse \
    --query "What is it?"

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