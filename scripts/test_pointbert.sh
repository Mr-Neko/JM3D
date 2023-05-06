#!/bin/bash

if [ -z "$1" ]; then
echo "Please provide a *.pt file as input"
exit 1
fi

model_file=$1
output_dir=/root/ULIP/mm_outputs/pointbert_8kpts_2e-4_ln

# CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PointBERT --npoints 1024 --validate_dataset_name scanobjectnn --all_image --output-dir $output_dir --evaluate_3d --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PointBERT --npoints 8192 --validate_dataset_name modelnet40 --all_image --output-dir $output_dir --evaluate_3d --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt