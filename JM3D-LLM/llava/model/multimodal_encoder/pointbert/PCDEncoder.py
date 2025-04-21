import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .dvae import Group
from .dvae import Encoder
from .logger import print_log
from typing import Optional, List, Union, Tuple
from easydict import EasyDict

from collections import OrderedDict
import os
import yaml
from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

# class PCDEncoderConfig(LlamaConfig):
#     model_type = 'pcd_encoder'

# class PCDEncoder(LlamaModel):
#     config_class = PCDEncoderConfig
class PCDEncoder(nn.Module):

    def __init__(self, config):
        # PointBERT as the backbone
        super(PCDEncoder, self).__init__()
        point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_base_8192point") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
        point_bert_config_addr = os.path.join(os.path.dirname(__file__), f"{point_bert_config_name}.yaml")
        print(f"Loading PointBERT config from {point_bert_config_addr}.")
        point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
        self.point_bert_config = point_bert_config
        if getattr(config, "use_color", False):
            point_bert_config.model.point_dims = 6
        use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
        self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
        
        self.point_backbone_config = {
            # TODO(coco): point_dims --> use_color
            # "point_cloud_dim": point_bert_config.model.point_dims,
            "point_cloud_dim": 6,
            "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
            "project_output_dim": config.hidden_size,
            "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
            "mm_use_im_start_end": config.mm_use_im_start_end,
            "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
            "use_max_pool": use_max_pool,
        }
        if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
            self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
        print_log(f"PointBERT config: {self.point_backbone_config}", logger="PCDEncoder")
        print_log(f"PointBERT point_token_len: {self.point_backbone_config['point_token_len']}", logger="PCDEncoder")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        print_log(f"Point backbone output dim: {backbone_output_dim}.")
        print_log(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        # if self.point_backbone_config['projection_hidden_layer'] > 0:
        #     # Add projection layer with linear layers and GELU activation
        #     projection_layers = []
        #     last_dim = backbone_output_dim
        #     for i in range(point_bert_config.model.projection_hidden_layer):
        #         projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
        #         projection_layers.append(nn.GELU())
        #         last_dim = self.point_backbone_config["projection_hidden_dim"][i]

        #     projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
        #     self.point_proj = nn.Sequential(*projection_layers)
        #     print_log.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        # else:
        #     # Single layer
        #     self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        # print_log.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.resampler = None 
        # self.resampler = Resampler(256, backbone_output_dim, 8)

        # self.fix_pointnet = False
        # self.fix_llm = False
    
    def load_checkpoint(self, bert_ckpt_path):
        self.point_backbone.load_checkpoint(bert_ckpt_path)


    def forward(
        self,
        pcs: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # import pdb
        # pdb.set_trace()
        # HACK: replace back original embeddings for LLaVA pretraining
        # orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        # point_backbone = getattr(self, 'point_backbone', None)
        # point_backbone_config = getattr(self, 'point_backbone_config', None)

        # if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and pcs is not None:
            # * enter when training or the first generation step of inference
        # with torch.no_grad() if self.fix_pointnet else nullcontext():
        #     if self.fix_pointnet:
        #         self.point_backbone.eval()
        self.point_backbone.eval()
        if type(pcs) is list:
            # * variable numbers of points
            point_features = []
            for point_cloud in pcs: # * iterate over batch
                point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                point_features.append(point_feature)
        else:
            point_features = self.point_backbone(pcs)

        if self.resampler is not None:
            point_features = self.resampler(point_features)
        return point_features

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     point_clouds: Optional[torch.FloatTensor] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:

    #     # HACK: replace back original embeddings for LLaVA pretraining
    #     orig_embeds_params = getattr(self, 'orig_embeds_params', None)

    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids)

    #     point_backbone = getattr(self, 'point_backbone', None)
    #     point_backbone_config = getattr(self, 'point_backbone_config', None)

    #     if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
    #         # * enter when training or the first generation step of inference
    #         with torch.no_grad() if self.fix_pointnet else nullcontext():
    #             if self.fix_pointnet:
    #                 self.point_backbone.eval()
    #             if type(point_clouds) is list:
    #                 # * variable numbers of points
    #                 point_features = []
    #                 for point_cloud in point_clouds: # * iterate over batch
    #                     point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
    #                     point_features.append(point_feature)
    #             else:
    #                 point_features = self.point_backbone(point_clouds)

    #         if self.resampler is not None:
    #             point_features = self.resampler(point_features)

    #         if type(point_clouds) is list:
    #             point_features = [self.point_proj(point_feature) for point_feature in point_features]
    #         else:
    #             point_features = self.point_proj(point_features)

    #         if self.resampler is None:
    #             dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    #         else:
    #             dummy_point_features = torch.zeros(256, point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    #         dummy_point_features = self.point_proj(dummy_point_features)

    #         new_input_embeds = []
    #         cur_point_idx = 0
    #         for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
    #             if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
    #                 # multimodal LLM, but the current sample is not multimodal
    #                 cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * seems doing nothing
    #                 new_input_embeds.append(cur_input_embeds)
    #                 cur_point_idx += 1
    #                 continue
    #             cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
    #             num_patches = cur_point_features.shape[0] # * number of point tokens
    #             if point_backbone_config['mm_use_point_start_end']:
    #                 if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
    #                     raise ValueError("The number of point start tokens and point end tokens should be the same.")
    #                 point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
    #                 for point_start_token_pos in point_start_tokens:
    #                     if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
    #                         raise ValueError("The point end token should follow the image start token.")
    #                     if orig_embeds_params is not None: # * will not update the original embeddings except for IMAGE_START_TOKEN and IMAGE_END_TOKEN
    #                         cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
    #                     else:
    #                         cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
    #                     cur_point_idx += 1
    #                 new_input_embeds.append(cur_new_input_embeds)
    #             else:
    #                 if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
    #                     raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
    #                 masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
    #                 mask_index_start = masked_indices[0]
    #                 if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
    #                     raise ValueError("The image patch tokens should be consecutive.")
    #                 if orig_embeds_params is not None:
    #                     cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
    #                 else:
    #                     cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
    #                 new_input_embeds.append(cur_new_input_embeds)
    #                 cur_point_idx += 1
    #         inputs_embeds = torch.stack(new_input_embeds, dim=0)

    #     return super(PointLLMLlamaModel, self).forward(
    #         input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds, use_cache=use_cache,
    #         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
    #         return_dict=return_dict
    #     )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config
        
        self.use_max_pool = use_max_pool # * whethet to max pool the features of different tokens

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # self.point_dims = config.point_dims
        # TODO(coco): point_dims --> use_color
        self.point_dims = 6
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.point_encoder.'):
                state_dict[k.replace('module.point_encoder.', '')] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print_log("PointBERT's weights are successfully loaded from {}".format(bert_ckpt_path), logger='Transformer')

    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f # * B, 1, C(384 + 384)