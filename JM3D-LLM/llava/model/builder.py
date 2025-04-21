#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
# from llava.model.multimodal_encoder.pointmlp.pointMLP import pointMLP, Model

# from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype='float16'):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype
    if 'llava' in model_name.lower(): # or 'vicuna' in model_name.lower():
        # Load LLaVA model
    #     if 'lora' in model_name.lower() and model_base is not None:
    #         lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         print('Loading LLaVA from base model...')
    #         model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, torch_dtype=kwargs.torch_dtype, **kwargs)
    #         token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    #         if model.lm_head.weight.shape[0] != token_num:
    #             model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #             model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    #         print('Loading additional LLaVA weights...')
    #         if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
    #             non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    #         else:
    #             # this is probably from HF Hub
    #             from huggingface_hub import hf_hub_download
    #             def load_from_hf(repo_id, filename, subfolder=None):
    #                 cache_file = hf_hub_download(
    #                     repo_id=repo_id,
    #                     filename=filename,
    #                     subfolder=subfolder)
    #                 return torch.load(cache_file, map_location='cpu')
    #             non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    #         non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    #         if any(k.startswith('model.model.') for k in non_lora_trainables):
    #             non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    #         model.load_state_dict(non_lora_trainables, strict=False)

    #         from peft import PeftModel
    #         print('Loading LoRA weights...')
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print('Merging LoRA weights...')
    #         model = model.merge_and_unload()
    #         print('Model is loaded...')
    #     elif model_base is not None:
    #         # this may be mm projector only
    #         print('Loading LLaVA from base model...')
    #         if 'mpt' in model_name.lower():
    #             if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
    #                 shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    #             model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path)
    #             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    #         mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    #         mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    #         model.load_state_dict(mm_projector_weights, strict=False)
    #     else:
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             # model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs).cuda()
    #             model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=kwargs['torch_dtype']).cuda()
    # else:
    #     # Load language model
    #     if model_base is not None:
    #         # PEFT model
    #         from peft import PeftModel
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    #         print(f"Loading LoRA weights from {model_path}")
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print(f"Merging weights")
    #         model = model.merge_and_unload()
    #         print('Convert to FP16...')
    #         model.to(torch.float16)
    #     else:
    #         use_fast = False
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=kwargs['torch_dtype'], **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs).cuda()
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=kwargs['torch_dtype']).cuda()
    pc_processor = None

    if 'llava' in model_name.lower(): # or 'vicuna' in model_name.lower():
        # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        # if mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        # if mm_use_im_start_end:
        #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        # model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_model().vision_tower
        # if vision_tower.device.type == 'meta':
        #     vision_tower = Model(points=1024, embed_dim=64, groups=1, res_expansion=1.0,
        #         activation="relu", bias=False, use_xyz=False, normalize="anchor",
        #         dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
        #         k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], ckpt_path='backbones/pointmlp/pointmlp_backbone.pt').cuda()
        #     model.get_model().vision_tower = vision_tower
        # else:
        #     import pdb
        #     pdb.set_trace()
        #     # NOTE: load the init weights when training stage 1
        #     # vision_tower.load_model_from_ckpt('backbones/pointmlp/pointmlp_backbone.pt')
        #     vision_tower = vision_tower.cuda()
        vision_tower = vision_tower.cuda()
        

        
        # try:
        #     mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        # except:
        #     mm_projector_weights = torch.load(os.path.join(model_path.replace(model_path.split('/')[-1],''), 'mm_projector', 'checkpoint-'+'4800'+'.bin'), map_location='cpu')
        # nofreeze = False
        # for k, v in mm_projector_weights.items():
        #     if 'vision_tower' in k:
        #         nofreeze = True
        #         break
        # if nofreeze:
        #     print('===>Loading unfreezed vision_tower weights from', model_path)
        #     vision_tower_weights = {k.replace('model.vision_tower.', ''): v for k, v in mm_projector_weights.items() if 'vision_tower' in k}
        #     vision_tower.load_state_dict(vision_tower_weights, strict=True)
        


        # # if not vision_tower.is_loaded:
        # #     vision_tower.load_model()
        # # vision_backbone = torch.load('backbones/pointmlp/pointmlp_backbone.pt', map_location='cpu')
        # # vision_tower.load_state_dict(vision_backbone['state_dict'], strict=True)
        # vision_tower.to(device='cuda', dtype=torch.float16)
        # # batch norm is required to be float32
        # for layer in vision_tower.modules():
        #     if isinstance(layer, torch.nn.BatchNorm1d):
        #         layer.float()
        # # pc_processor = vision_tower.pc_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, pc_processor, context_len
