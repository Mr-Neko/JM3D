# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

import wandb
import os
# wandb.init(
#     project="JM3D_refactor",
#     name='jm3d_stage1_rank_' + os.environ['LOCAL_RANK'],
# )

# wandb.init(
#     project="JM3D_refactor",
#     name='jm3d_stage2_rank_' + os.environ['LOCAL_RANK'],
# )

wandb.init(
    project="JM3D_refactor",
    name='jm3d_stage3_rank_' + os.environ['LOCAL_RANK'],
)

if __name__ == "__main__":
    print('os.environ[LOCAL_RANK] ', os.environ['LOCAL_RANK'])
    train()
