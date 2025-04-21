# JM3D-LLM

*3D visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

## Contents
- [Data Preparation](#data-preparation)
- [Install](#install)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)

## Data Preparation

Dwonload the instuction data [pc_chat_Cap3D_660k.json](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/EklPiEX2CTZBmPrLVpotyaUBtKhmtxos7RI4I66Ld7eYzw?e=vBYPAi), and put it into `data/Objaverse/`.

Download the [Cap3D datasets](https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips), and unzip and put the point cloud data into `data/Objaverse/Cap3D_pcs_pt/`.

## Install

1. Following the LLaVA settings, clone the repository and navigate to LLaVA folder
```bash
git clone https://https://github.com/Mr-Neko/JM3D.git
cd JM3D/JM3D-LLM
```

2. Install Package
```Shell
conda create -n jm3d-llm python=3.10 -y
conda activate jm3d-llm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn==1.0.2
```

## Fine-tuning
### Backbones
#### Visual Backbone
Download the JM3D-pretrained [PointMLP checkpoint](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/EklPiEX2CTZBmPrLVpotyaUBtKhmtxos7RI4I66Ld7eYzw?e=vBYPAi), and put it into `backbones/pointmlp/`

#### Language Backbone
Download the [Vicuna-7B checkpoint](https://huggingface.co/lmsys/vicuna-7b-v1.1), and put it into `backbones/vicuna-7b/`

### Running
```
bash scripts/pretrain_pc.sh
```

## Evaluation
Modify the `model_path` in `scripts/test_pretrain_pc.sh`, and 
```
bash scripts/test_pretrain_pc.sh
```

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
