#!/bin/bash

if [ -z "$1" ]; then
echo "Please provide a *.pt file as input"
exit 1
fi

model_file=$1
output_dir=/root/ULIP/mm_outputs/pointmlp_8kpts

CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PN_MLP --npoints 1024 --validate_dataset_name scanobjectnn --output-dir $output_dir --all_image --evaluate_3d --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PN_MLP --npoints 8192 --validate_dataset_name modelnet40 --output-dir $output_dir --all_image --evaluate_3d --test_ckpt_addr $model_file 2>&1 | tee $output_dir/log.txt