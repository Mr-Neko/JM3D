# MODEL_PATH='/home/myw/wuchangli/PointLLM/outputs/PointLLM_train_stage1/colorv1.1_v5'
MODEL_PATH='/home/myw/wuchangli/yk/JM3D/JM3D-LLM/checkpoints/stage1_ckpt'
MODEL_PATH='/home/myw/wuchangli/yk/JM3D/JM3D-LLM/checkpoints/stage2_ckpt'


CUDA_VISIBLE_DEVICES=2 \
    python llava/eval/run_llava_chat.py \
    --model_path $MODEL_PATH \
    --data_path data/objaverse_data \
    --torch_dtype float32