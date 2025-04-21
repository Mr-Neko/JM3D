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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .multimodal_encoder.builder import build_vision_tower
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(LlavaLlamaModel, self).__init__(config)
        # if hasattr(config, "mm_vision_tower"):
        #     # HACK: for FSDP
        #     self.vision_tower = [build_vision_tower(config.mm_vision_tower, delay_load=True, multi_token=True)]
        #     # self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
        #     # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        # if hasattr(config, "use_mm_proj"):
        #     self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        self.vision_tower = build_vision_tower(config)
        # self.mm_projector = 

        self.point_backbone_config = self.vision_tower.point_backbone_config
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = self.point_backbone_config['backbone_output_dim']
            for i in range(self.vision_tower.point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.mm_projector = nn.Sequential(*projection_layers)
            # logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.mm_projector = nn.Linear(self.point_backbone_config['backbone_output_dim'], self.point_backbone_config['project_output_dim'])
        # logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.resampler = None 
        # self.resampler = Resampler(256, backbone_output_dim, 8)

        # TODO(coco): seems no need
        # self.fix_pointnet = False
        # self.fix_llm = False


    # def initialize_vision_modules(self, vision_tower, mm_vision_select_layer=None,
    #                               pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
    #     import pdb
    #     pdb.set_trace()
    #     self.config.mm_vision_tower = vision_tower

    #     # image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    #     # if not hasattr(self, 'vision_tower'):
    #     #     vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
    #     # else:
    #     #     vision_tower = self.vision_tower[0]
    #     vision_tower = build_vision_tower(vision_tower, multi_token=True)
    #     print('Not Freezing Vision Tower')
    #     vision_tower.requires_grad_(True)
    #     # vision_tower = vision_tower.to(torch.float16)
    #     self.vision_tower = vision_tower

    #     # num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

    #     self.config.use_mm_proj = True
    #     self.config.mm_hidden_size = vision_tower.hidden_size

    #     if not hasattr(self, 'mm_projector'):
    #         self.mm_projector = nn.Linear(vision_tower.hidden_size, self.config.hidden_size)

    #     if pretrain_mm_mlp_adapter is not None:
    #         mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
    #         self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

    #     return dict(
    #         # image_processor=image_processor,
    #         image_token_len=1,
    #         # vision_config=vision_config
    #     )
    def encode_images(self, images):
        image_features = self.vision_tower(pcs=images)
        image_features = self.mm_projector(image_features)
        return image_features
    
    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.vision_tower.load_checkpoint(checkpoint_path)


    def prepare_inputs_labels_for_multimodal(
        # self, input_ids, attention_mask, past_key_values, labels, images
        self, input_ids, attention_mask, past_key_values, images
    ):
        vision_tower = self.vision_tower
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     # print("Warning: vision_tower is None or images is None or input_ids.shape[1] == 1")
        #     # print(vision_tower, images.shape, input_ids.shape)
        #     if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
        #         attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
        #     return input_ids, attention_mask, past_key_values, None, labels
        image_features = self.encode_images(images)
        # print('=======image_features:', image_features.shape)
        # image_features = image_features.half() # if self.get_model().config.fp16 else image_features
        new_input_embeds = []
        # new_labels = [] if labels is not None else None
        cur_image_idx = 0

        if self.resampler is None:
            dummy_point_features = torch.zeros(self.point_backbone_config['point_token_len'], self.point_backbone_config['backbone_output_dim'], device=image_features.device, dtype=image_features.dtype)
        else:
            dummy_point_features = torch.zeros(256, self.point_backbone_config['backbone_output_dim'], device=image_features.device, dtype=image_features.dtype)

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        # new_input_embeds = []
        # cur_point_idx = 0
        # for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
        #     if (cur_input_ids == self.point_backbone_config['point_patch_token']).sum() == 0:
        #         # multimodal LLM, but the current sample is not multimodal
        #         cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * seems doing nothing
        #         new_input_embeds.append(cur_input_embeds)
        #         cur_point_idx += 1
        #         continue
        #     cur_point_features = image_features[cur_point_idx].to(device=cur_input_embeds.device)
        #     num_patches = cur_point_features.shape[0] # * number of point tokens
        #     if self.point_backbone_config['mm_use_im_start_end']:
        #         if (cur_input_ids == self.point_backbone_config["point_start_token"]).sum() != (cur_input_ids == self.point_backbone_config["point_end_token"]).sum():
        #             raise ValueError("The number of point start tokens and point end tokens should be the same.")
        #         point_start_tokens = torch.where(cur_input_ids == self.point_backbone_config["point_start_token"])[0]
        #         for point_start_token_pos in point_start_tokens:
        #             if cur_input_ids[point_start_token_pos + num_patches + 1] != self.point_backbone_config["point_end_token"]:
        #                 raise ValueError("The point end token should follow the image start token.")
        #             if orig_embeds_params is not None: # * will not update the original embeddings except for IMAGE_START_TOKEN and IMAGE_END_TOKEN
        #                 cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
        #             else:
        #                 cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
        #             cur_point_idx += 1
        #         new_input_embeds.append(cur_new_input_embeds)
        #     else:
        #         if (cur_input_ids == self.point_backbone_config["point_patch_token"]).sum() != num_patches:
        #             raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
        #         masked_indices = torch.where(cur_input_ids == self.point_backbone_config["point_patch_token"])[0]
        #         mask_index_start = masked_indices[0]
        #         if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
        #             raise ValueError("The image patch tokens should be consecutive.")
        #         if orig_embeds_params is not None:
        #             cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
        #         else:
        #             cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
        #         new_input_embeds.append(cur_new_input_embeds)
        #         cur_point_idx += 1
        # inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # return None, attention_mask, past_key_values, inputs_embeds, new_labels
        # import pdb
        # pdb.set_trace()
        inputs_embeds = self.embed_tokens(input_ids)
        # import pdb
        # pdb.set_trace()
        # print(input_ids.shape, image_features.shape)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_embeds = inputs_embeds[batch_idx]
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # print('the current sample is not multimodal', cur_input_ids)
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = cur_input_embeds + (0. * self.mm_projector(dummy_point_features)).sum()
                new_input_embeds.append(cur_input_embeds)
                # if labels is not None:
                #     new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            cur_new_input_embeds = []
            # if labels is not None:
            #     cur_labels = labels[batch_idx]
            #     cur_new_labels = []
            #     assert cur_labels.shape == cur_input_ids.shape

            # NOTE(coco): mm_use_im_start_end
            cur_point_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
            num_patches = cur_point_features.shape[0]

            if self.point_backbone_config['mm_use_im_start_end']:
                point_start_tokens = torch.where(cur_input_ids == self.point_backbone_config["point_start_token"])[0]
                for point_start_token_pos in point_start_tokens:
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                    cur_image_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            else:
                if (cur_input_ids == self.point_backbone_config["point_patch_token"]).sum() != num_patches:
                    raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                masked_indices = torch.where(cur_input_ids == self.point_backbone_config["point_patch_token"])[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                if orig_embeds_params is not None:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                else:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                cur_image_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)


            # image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # print(image_token_indices)
            # # insert pc feature
            # while image_token_indices.numel() > 0:
            #     cur_image_features = image_features[cur_image_idx].unsqueeze(0) if image_features.ndim == 2 else image_features[cur_image_idx]
            #     # print('======cur_image_features:', cur_image_features.shape)
            #     image_token_start = image_token_indices[0]
            #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[:image_token_start-1]).detach())
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
            #         cur_new_input_embeds.append(cur_image_features)
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
            #         if labels is not None:
            #             cur_new_labels.append(cur_labels[:image_token_start])
            #             cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
            #             cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
            #             cur_labels = cur_labels[image_token_start+2:]
            #     else:
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[:image_token_start]))
            #         cur_new_input_embeds.append(cur_image_features)
            #         if labels is not None:
            #             cur_new_labels.append(cur_labels[:image_token_start])
            #             cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
            #             # print('======cur_image_features:', cur_image_features.shape)
            #             cur_labels = cur_labels[image_token_start+1:]
            #     cur_image_idx += 1
            #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            #         cur_input_ids = cur_input_ids[image_token_start+2:]
            #     else:
            #         cur_input_ids = cur_input_ids[image_token_start+1:]
            #     image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # if cur_input_ids.numel() > 0:
            #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids).detach())
            #     else:
            #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids))
            #     if labels is not None:
            #         cur_new_labels.append(cur_labels)
            # cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # # print([x.shape for x in cur_new_input_embeds])
            # # print([x.isnan().sum() for x in cur_new_input_embeds])
            # # [32, 512], [512], [24, 512]
            # cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # new_input_embeds.append(cur_new_input_embeds)
            # if labels is not None:
            #     cur_new_labels = torch.cat(cur_new_labels, dim=0)
            #     new_labels.append(cur_new_labels)

        # if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        #     max_len = max(x.shape[0] for x in new_input_embeds)

        #     new_input_embeds_align = []
        #     for cur_new_embed in new_input_embeds:
        #         cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
        #         new_input_embeds_align.append(cur_new_embed)
        #     new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        #     if labels is not None:
        #         new_labels_align = []
        #         _new_labels = new_labels
        #         for cur_new_label in new_labels:
        #             cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
        #             new_labels_align.append(cur_new_label)
        #         new_labels = torch.stack(new_labels_align, dim=0)

        #     if attention_mask is not None:
        #         new_attention_mask = []
        #         for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
        #             new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
        #             new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
        #             cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
        #             new_attention_mask.append(cur_new_attention_mask)
        #         attention_mask = torch.stack(new_attention_mask, dim=0)
        #         assert attention_mask.shape == new_labels.shape
        # else:
        #     # for cur_new_embed in new_input_embeds: print(cur_new_embed.dtype, cur_new_embed.shape)
        #     new_input_embeds = torch.stack(new_input_embeds, dim=0)
        #     if labels is not None:
        #         new_labels  = torch.stack(new_labels, dim=0)

        #     if attention_mask is not None:
        #         new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
        #         attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
        #         assert attention_mask.shape == new_input_embeds.shape[:2]

        # print(new_input_embeds.dtype, new_input_embeds.shape)
        # return None, attention_mask, past_key_values, inputs_embeds, new_labels
        return None, attention_mask, past_key_values, inputs_embeds

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     pcs: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:

    #     # HACK: replace back original embeddings for LLaVA pretraining
    #     orig_embeds_params = getattr(self, 'orig_embeds_params', None)
    #     # if orig_embeds_params is not None:
    #     #     orig_embeds_params = orig_embeds_params[0]
    #     #     with torch.no_grad():
    #     #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

    #     # if inputs_embeds is None:
    #     #     inputs_embeds = self.embed_tokens(input_ids)

    #     # vision_tower = getattr(self, 'vision_tower', None)
    #     # if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
    #     #     # TODO: this is a modified multimodal LLM -- Haotian Liu
    #     #     vision_tower = vision_tower[0]  # HACK: for FSDP
    #     #     with torch.no_grad():
    #     #         # if type(images) is list:
    #     #         #     # variable length images
    #     #         #     image_features = []
    #     #         #     for image in images:
    #     #         #         image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
    #     #         #         select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
    #     #         #         select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
    #     #         #         image_feature = select_hidden_state[:, 1:]
    #     #         #         image_features.append(image_feature)
    #     #         # else:
    #     #         #     image_forward_outs = vision_tower(images, output_hidden_states=True)
    #     #         #     select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
    #     #         #     select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
    #     #         #     image_features = select_hidden_state[:, 1:]
    #     #         image_features = self.encode_images(images)

    #     #     dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    #     #     dummy_image_features = self.mm_projector(dummy_image_features)

    #     #     new_input_embeds = []
    #     #     cur_image_idx = 0
    #     #     for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
    #     #         if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
    #     #             # multimodal LLM, but the current sample is not multimodal
    #     #             cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
    #     #             new_input_embeds.append(cur_input_embeds)
    #     #             cur_image_idx += 1
    #     #             continue
    #     #         if vision_tower.config.use_im_start_end:
    #     #             cur_image_features = image_features[cur_image_idx]
    #     #             num_patches = cur_image_features.shape[0]
    #     #             if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
    #     #                 raise ValueError("The number of image start tokens and image end tokens should be the same.")
    #     #             image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
    #     #             for image_start_token_pos in image_start_tokens:
    #     #                 cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
    #     #                 num_patches = cur_image_features.shape[0]
    #     #                 if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
    #     #                     raise ValueError("The image end token should follow the image start token.")
    #     #                 if orig_embeds_params is not None:
    #     #                     cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
    #     #                 else:
    #     #                     cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
    #     #                 cur_image_idx += 1
    #     #             new_input_embeds.append(cur_new_input_embeds)
    #     #         else:
    #     #             cur_image_features = image_features[cur_image_idx]
    #     #             num_patches = cur_image_features.shape[0]
    #     #             if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() != num_patches:
    #     #                 raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
    #     #             masked_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    #     #             mask_index_start = masked_indices[0]
    #     #             if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
    #     #                 raise ValueError("The image patch tokens should be consecutive.")
    #     #             if orig_embeds_params is not None:
    #     #                 cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
    #     #             else:
    #     #                 cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
    #     #             new_input_embeds.append(cur_new_input_embeds)
    #     #             cur_image_idx += 1
    #     #     inputs_embeds = torch.stack(new_input_embeds, dim=0)
    #     # try:
    #     #     print('before pro: ', input_ids.shape, inputs_embeds, pcs.shape)
    #     # except:
    #     #     print('before pro: ', input_ids.shape, inputs_embeds)
    #     input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, pcs)
    #     # try:
    #     #     print('after pro: ', input_ids, inputs_embeds.shape)
    #     # except:
    #     #     print('after pro: ', input_ids.shape, inputs_embeds)
    #     return super(LlavaLlamaModel, self).forward(
    #         input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds, use_cache=use_cache,
    #         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
    #         return_dict=return_dict
    #     ), labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pcs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
 
        # input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, pcs)
        input_ids, attention_mask, past_key_values, inputs_embeds = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, pcs)
        # try:
        #     print('after pro: ', input_ids, inputs_embeds.shape)
        # except:
        #     print('after pro: ', input_ids.shape, inputs_embeds)
        # import pdb
        # pdb.set_trace()
        return super(LlavaLlamaModel, self).forward(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

logits_list = []

class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pcs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print('-'*50)
        # print('input ids:')
        # print(input_ids)
        # print('-'*50)


        # import pdb
        # pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs, labels = self.model(
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pcs=pcs,
            # labels=labels,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # logits_list.append(logits.detach().float().cpu().numpy())
        # if (len(logits_list) % 10 == 0):
        #     import pdb
        #     pdb.set_trace()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pcs": kwargs.get("pcs", None),
            }
        )
        return model_inputs

    # def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
    #                                 tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
    #     vision_config = self.get_model().vision_tower[0].config
    #     vision_config.use_im_start_end = mm_use_im_start_end
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #     self.resize_token_embeddings(len(tokenizer))

    #     if mm_use_im_start_end:
    #         num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #         self.resize_token_embeddings(len(tokenizer))
    #         vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    #         if num_new_tokens > 0:
    #             input_embeddings = self.get_input_embeddings().weight.data
    #             output_embeddings = self.get_output_embeddings().weight.data

    #             input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)
    #             output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #                 dim=0, keepdim=True)

    #             input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #             output_embeddings[-num_new_tokens:] = output_embeddings_avg

    #         if tune_mm_mlp_adapter:
    #             self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
    #             for p in self.get_input_embeddings().parameters():
    #                 p.requires_grad = True
    #             for p in self.get_output_embeddings().parameters():
    #                 p.requires_grad = False

    #         if pretrain_mm_mlp_adapter:
    #             mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
    #             embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
    #             assert num_new_tokens == 2
    #             if input_embeddings.shape == embed_tokens_weight.shape:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
    #             elif embed_tokens_weight.shape[0] == num_new_tokens:
    #                 input_embeddings[-num_new_tokens:] = embed_tokens_weight
    #             else:
    #                 raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

    #     vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_im_start_end = point_backbone_config['mm_use_im_start_end'] = config.mm_use_im_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_im_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, training_stage=1):
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_im_start_end = point_backbone_config['mm_use_im_start_end'] = config.mm_use_im_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_im_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if training_stage == 1:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")



# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
