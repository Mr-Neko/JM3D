from .clip_encoder import CLIPVisionTower
# from .pointmlp.pointMLP import pointMLP, Model
from .pointbert.PCDEncoder import PCDEncoder
import torch

# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
#         return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_vision_tower(config):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
    #     return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # raise ValueError(f'Unknown vision tower: {vision_tower}')
    # vision_tower = pointMLP()
    # if kwargs['multi_token']: print('Using multi-token visual embeddings!!!!!!!')
    # print('vision_tower_cfg', vision_tower_cfg)
    return PCDEncoder(config)


#     return Model(points=1024, embed_dim=64, groups=1, res_expansion=1.0,
#             activation="relu", bias=False, use_xyz=False, normalize="anchor",
#             dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
#             k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], ckpt_path=vision_tower_cfg, **kwargs)



    # vision_tower = vision_tower.to('cuda:0')
    # for k in vision_tower.state_dict().keys():
    #     print(k, vision_tower.state_dict()[k].shape)
    # print(vision_tower_cfg.mm_vision_tower)

    # if hasattr(vision_tower_cfg, 'mm_vision_tower'):
    #     backbone = torch.load(vision_tower_cfg.mm_vision_tower, map_location='cpu')
    #     print(f"Loading vision tower from {vision_tower_cfg.mm_vision_tower}")
    # else:
    #     backbone = torch.load(vision_tower_cfg.vision_tower, map_location='cpu')
    #     print(f"Loading vision tower from {vision_tower_cfg.vision_tower}")

    # for k in vision_tower.state_dict().keys():
    #     if k not in backbone['state_dict']:
    #         print(f"Missing key {k} in pretrained vision tower")
    #     else:
    #         print(f"Loaded key {k} from pretrained vision tower", vision_tower.state_dict()[k].shape, backbone['state_dict'][k].shape)
    # vision_tower.load_state_dict(backbone['state_dict'], strict=True)
    # print(f"Loaded vision tower from {vision_tower_cfg.vision_tower}")
    return vision_tower