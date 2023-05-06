'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
from torch.nn import functional as F
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import  *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        # self.class_layer = nn.Linear(512, kwargs.class_num)
        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None, degrees=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)

        # class_output = F.relu(self.class_layer(pc_embed))
        # class_output = F.softmax(class_output, dim=-1)

        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'class_output': 0, 
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}

class ALL_IMAGE_TRANSFORMER(nn.Module):
    def __init__(self, width=512, layers=1, heads=8, image_nums=1) -> None:
        super(ALL_IMAGE_TRANSFORMER, self).__init__()

        self.degree_embedding = nn.Parameter(torch.rand(30, width))

        self.depth_embedding = nn.Parameter(torch.rand(2, width))

        self.ln_final = nn.LayerNorm(width)
    
    def forward(self, image_features, degrees, text_embed):
        x = image_features

        bsz = image_features.shape[0]
        degree_embedding = []
        
        for b in range(bsz):
            degree_embedding.append(self.degree_embedding[degrees[b]])

        depth_embedding = torch.cat((self.depth_embedding[0].unsqueeze(dim=0).repeat(degrees.shape[1] // 2, 1), 
            self.depth_embedding[1].unsqueeze(dim=0).repeat(degrees.shape[1] // 2, 1)), dim=0).unsqueeze(dim=0)
        degree_embedding = torch.stack(degree_embedding,dim=0)
        x = image_features + degree_embedding + depth_embedding

        x = self.ln_final(x)
        return x



class ULIP_WITH_ALL_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        # super().__init__(ssl_mlp_dim, ssl_emb_dim, **kwargs)
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.args = kwargs.args
        self.strength_image = ALL_IMAGE_TRANSFORMER()

        if self.args.class_text != 'None':
            self.class_layer = nn.Linear(512, kwargs.class_num)

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x
    
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None, degress=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)

        if self.args.class_text != 'None':
            class_output = F.relu(self.class_layer(pc_embed))
            class_output = F.softmax(class_output, dim=-1)
        else:
            class_output = 0
        
        if image is not None:
            image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
            image_embed = self.encode_image(image)
            image_embed = image_embed.view(pc_embed.shape[0], -1, image_embed.shape[-1])
            strength_image_embed = self.strength_image(image_embed, degress, text_embed_all)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': strength_image_embed,
                    'origin_image_embed': image_embed,
                    'class_output': class_output, 
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}

def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model, args):

    if args.soft_label:

        if args.class_text != 'None':
            return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc', 'soft_loss', 'class_loss']
        else:
            return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc', 'soft_loss']
    
    else:
        if args.class_text != 'None':
            return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc', 'class_loss']
        else:
            return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']   



def ULIP_PN_SSG(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    point_encoder = Pointnet2_Ssg()
    class_num = 55
    if args.class_text == 'sub':
        class_num = 205
    pc_feat_dims = 256
    # =====================================================================

    if args.all_image:
        model = ULIP_WITH_ALL_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                          transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims, class_num=class_num, args=args)
    
    else:
        model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PN_MLP(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointmlp.pointMLP import pointMLP
    point_encoder = pointMLP()
    class_num = 55
    if args.class_text == 'sub':
        class_num = 205
    pc_feat_dims = 256
    # =====================================================================
    if args.all_image:
        model = ULIP_WITH_ALL_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims, class_num=class_num, args=args)
    else:   
        model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims, class_num=class_num)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model

def ULIP_PointBERT(args):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)

    # =====================================================================
    # import the 3D backbone and specify the output point cloud feature dimension
    from models.pointbert.point_encoder import PointTransformer
    config_addr = './models/pointbert/PointTransformer_8192point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    class_num = 55
    if args.class_text == 'sub':
        class_num = 205
    pc_feat_dims = 768
    # =====================================================================
    if args.all_image:
        model = ULIP_WITH_ALL_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims, class_num=class_num, args=args)

    else:    
        model = ULIP_WITH_IMAGE(embed_dim=512, vision_width=768, point_encoder=point_encoder, vision_model=vision_model,
                            context_length=77, vocab_size=49408,
                            transformer_width=512, transformer_heads=8, transformer_layers=12, pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        # load the pretrained model
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

    return model