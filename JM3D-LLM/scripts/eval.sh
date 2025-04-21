# MODEL_PATH='checkpoints/stage2_ckpt'
MODEL_PATH='checkpoints/stage3_ckpt'
# export MODEL_PATH=/home/myw/wuchangli/PointLLM/checkpoints/PointLLM_7B_v1.1
export RES_PATH=$MODEL_PATH/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json

python llava/eval/eval_objaverse.py --model_name $MODEL_PATH --task_type captioning --prompt_index 2 --use_color

python llava/eval/traditional_evaluator.py --results_path $RES_PATH