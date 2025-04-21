# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import copy
from dataclasses import dataclass, field
from typing import Optional, List
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import torch.nn as nn
import numpy as np

# TODO: import and use code from ../data/dataset.py

from llava.constants import IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    # mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=True)


@dataclass
class DataArguments:
    # data_path: str = field(default=None,
    #                        metadata={"help": "Path to the training data."})
    # anno_path: str = field(default=None,
    #                        metadata={"help": "Path to the anno data."})
    # lazy_preprocess: bool = False
    # is_multimodal: bool = False
    # # image_folder: Optional[str] = field(default=None)
    # pc_folder: Optional[str] = field(default=None)
    # uniform: bool = field(default=True)
    # augment: bool = field(default=True)
    # use_height: bool = field(default=False)
    # npoints: int = field(default=8192)
    data_path: str = field(default="data/objaverse_data", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=True, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    num_gpus: int = field(default=1)
    training_stage: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


# def preprocess_multimodal(
#     sources: Sequence[str],
#     multimodal_cfg: dict,
#     cur_token_len: int,
# ) -> Dict:
#     is_multimodal = multimodal_cfg.is_multimodal
#     # image_token_len = multimodal_cfg['image_token_len']
#     image_token_len = cur_token_len
#     if not is_multimodal:
#         return sources

#     for source in sources:
#         # if multimodal_cfg['sep_image_conv_front']:
#         assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#         source[0]['value'] = source[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
#         # source[0]['value'] = DEFAULT_IMAGE_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
#         # source[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*64 + DEFAULT_IM_END_TOKEN + conversation_lib.default_conversation.sep + source[0]['value']
#         # source[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*64 + DEFAULT_IM_END_TOKEN + conversation_lib.default_conversation.sep + source[0]['value']
#         source[0]['value'] = DEFAULT_IMAGE_TOKEN + conversation_lib.default_conversation.sep + source[0]['value']
#         # for sentence in source:
#         #     replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
#         #     if multimodal_cfg['use_im_start_end']:
#         #         replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#         #     sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#     return sources

def preprocess_multimodal(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
) -> Dict:
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    for source in sources:
        for sentence in source:
            replace_token = default_point_patch_token * point_token_len 
            # replace_token = default_point_patch_token * 256 # qformer 
            if point_backbone_config['mm_use_im_start_end']:
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)

    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print('sources: ', sources)
    # print('conversations: ', conversations)
    # exit()
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer, has_image)
    # if conversation_lib.default_conversation.version == "mpt":
    #     return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = json.load(open(data_path, "r"))

#         logging.warning("Formatting inputs...")
#         sources = [example["conversations"] for example in list_data_dict]
#         data_dict = preprocess(sources, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  anno_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  multimodal_cfg: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = json.load(open(anno_path, "r"))
#         self.pc_folder = multimodal_cfg.pc_folder

#         logging.warning("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.multimodal_cfg = multimodal_cfg
#         self.sample_points_num = multimodal_cfg.npoints
#         self.permutation = np.arange(multimodal_cfg.npoints)
#         self.uniform = True
#         self.augment = True        
#         # =================================================
#         # TODO: disable for backbones except for PointNEXT!!!
#         self.use_height = multimodal_cfg.use_height
#         # =================================================
#         self.point_indicator = '<point>'

#     def pc_norm(self, pc):
#         """ pc: NxC, return NxC """
#         centroid = np.mean(pc, axis=0)
#         pc = pc - centroid
#         m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#         pc = pc / m
#         return pc

#     def random_sample(self, pc, num):
#         np.random.shuffle(self.permutation)
#         pc = pc[self.permutation[:num]]
#         return pc
    
#     def __len__(self):
#         return len(self.list_data_dict)

#     def __getitem__(self, index) -> Dict[str, torch.Tensor]:
#         # import pdb
#         # pdb.set_trace()
#         # sources = self.list_data_dict[i]
#         # if isinstance(i, int):
#         #     sources = [sources]
#         # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         # if 'pc' in sources[0]:

#         #     pc_file = self.list_data_dict[i]['pc']
#         #     if pc_file.endswith('.pt'):
#         #         # pc_data = torch.load(os.path.join(self.pc_folder, pc_file))
#         #         pc_data_name = f'{pc_file[:-3]}_{self.sample_points_num}'
#         #         try:
#         #             pc_data = np.load(os.path.join(self.pc_folder, pc_file[:-3], pc_data_name + '.npz'))
#         #         except:
#         #             # pc_data = np.load(os.path.join(self.pc_folder, pc_file[:-3], pc_data_name + '.npy'))
#         #             assert('fail to load npz file !!!!!!!!!!!')

#         #         # pc_data = pc_data[:3,:].permute(1, 0).numpy().astype(np.float32)
#         #         pc_data = pc_data['arr_0']
#         #     else:
#         #         pc_data = IO.get(os.path.join(self.pc_folder, pc_file)).astype(np.float32)

#         #     if self.uniform and self.sample_points_num < pc_data.shape[0]:
#         #         pc_data = farthest_point_sample(pc_data, self.sample_points_num)
#         #     else:
#         #         pc_data = self.random_sample(pc_data, self.sample_points_num)
            
#         #     pc_data = self.pc_norm(pc_data)

#         #     if self.augment:
#         #         pc_data = random_point_dropout(pc_data[None, ...])
#         #         pc_data = random_scale_point_cloud(pc_data)
#         #         pc_data = shift_point_cloud(pc_data)
#         #         pc_data = rotate_perturbation_point_cloud(pc_data)
#         #         pc_data = rotate_point_cloud(pc_data)
#         #         pc_data = pc_data.squeeze()

#         #     if self.use_height:
#         #         self.gravity_dim = 1
#         #         height_array = pc_data[:, self.gravity_dim:self.gravity_dim + 1] - pc_data[:,
#         #                                                                 self.gravity_dim:self.gravity_dim + 1].min()
#         #         pc_data = np.concatenate((pc_data, height_array), axis=1)
#         #         pc_data = torch.from_numpy(pc_data).float()
#         #     else:
#         #         pc_data = torch.from_numpy(pc_data).float()
#         #     # cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)   # FIXME: 14 is hardcoded patch size
#         #     cur_token_len = 1   # pc token is only one token
#         #     # print('berfore process', sources)
#         #     sources = preprocess_multimodal(
#         #         copy.deepcopy([e["conversations"] for e in sources]),
#         #         self.multimodal_cfg, cur_token_len)
#         # else:
#         #     sources = copy.deepcopy([e["conversations"] for e in sources])
#         # # print('after process', sources)
#         # data_dict = preprocess(
#         #     sources,
#         #     self.tokenizer, 
#         #     has_image=True)
#         # if isinstance(i, int):
#         #     data_dict = dict(input_ids=data_dict["input_ids"][0],
#         #                      labels=data_dict["labels"][0])

#         # # print(self.list_data_dict[i])
#         # # print(pc_data.shape)
#         # # pc exist in the data
#         # if 'pc' in self.list_data_dict[i]:
#         #     data_dict['pc'] = pc_data
#         # elif self.data_args.is_multimodal:
#         #     # pc does not exist in the data, but the model is multimodal
#         #     data_dict['pc'] = torch.zeros(1024, 3)
#         # # # image exist in the data
#         # # if 'image' in self.list_data_dict[i]:
#         # #     data_dict['image'] = image
#         # # elif self.multimodal_cfg['is_multimodal']:
#         # #     # image does not exist in the data, but the model is multimodal
#         # #     crop_size = self.multimodal_cfg['image_processor'].crop_size
#         # #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         # return data_dict
#         sources = self.list_data_dict[index]
#         print(len(self.list_data_dict), sources)
#         import pdb
#         pdb.set_trace()
#         if isinstance(index, int):
#             sources = [sources]
#         assert len(sources) == 1, "sources should be a list"
#         if self.point_indicator in sources[0]['conversations'][0]['value']:

#             object_id = self.list_data_dict[index]['object_id']

#             # Point cloud representation
#             point_cloud = self._load_point_cloud(object_id) # * N, C
#             point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm

#             if self.tokenizer is None:
#                 data_dict = dict(
#                     point_clouds=torch.from_numpy(point_cloud.astype(np.float32)),
#                     object_ids=object_id
#                 )
#                 return data_dict

#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]), self.multimodal_cfg.point_backbone_config, point_indicator=self.point_indicator)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])

#         data_dict = preprocess_v1(
#             sources,
#             self.tokenizer)

#         if isinstance(index, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # point exist in the data
#         if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
#             data_dict['point_clouds'] = torch.from_numpy(point_cloud.astype(np.float32))

#         return data_dict
    
#     def _load_point_cloud(self, object_id):
#         filename = f"{object_id}_{self.sample_points_num}.npy"
#         try:
#             point_cloud = np.load(os.path.join(self.data_path, filename))
#             print(f'load from {os.path.join(self.data_path, filename)}')
#         except:
#             pc_fn = os.path.join('/home/myw/haowei/ULIP/data/ULIP-Objaverse_triplets/objaverse_pc_parallel', object_id, filename)
#             print(f'load from {pc_fn}')
#             try:
#                 point_cloud = np.load(pc_fn)
#             except:
#                 pc_fn = pc_fn.replace('.npy','.npz')
#                 point_cloud = np.load(pc_fn)['arr_0']

#             # copy to the data_path
#             # np.save(os.path.join(self.data_path, filename), point_cloud)

#         # TODO(coco): use_color default True
#         # if not self.use_color:
#         #     try:
#         #         point_cloud = point_cloud[:, :3]
#         #     except:
#         #         print(object_id, point_cloud.shape)
#         #         # raise path
#         #         raise ValueError(object_id + ' point cloud shape error')

#         return point_cloud


class LazySupervisedDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 tokenizer=None,
                 pointnum=8192,
                 split='train',
                 conversation_types=None, # * default is simple_des, used for stage1 pre-train
                 use_color=True,
                 data_args=None):

        """
        split: only considered when data_args.split_train_val is True.
        conversation_types: tuple, used to filter the data, default is ('simple_description'), other types is:
            "detailed_description", "single_round", "multi_round".
        tokenizer: load point clouds only if None
        """
        super(LazySupervisedDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split 
        if conversation_types is None:
            self.conversation_types = ("simple_description",)
        else:
            self.conversation_types = conversation_types

        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color

        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = '<point>'

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)
        
        # * print the conversations_type
        print(f"Using conversation_type: {self.conversation_types}") 
        # * print before filtering
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")

        # * iterate the list and filter
        # * these two ids have corrupted colored point files, so filter them when use_color is True
        filter_ids = ['6760e543e1d645d5aaacd3803bcae524', 'b91c0711149d460a8004f9c06d3b7f38'] if self.use_color else []

        # Iterate the list, filter those "conversation_type" not in self.conversation_types
        self.list_data_dict = [
            data for data in self.list_data_dict 
            if data.get('conversation_type', 'simple_description') in self.conversation_types 
            and data.get('object_id') not in filter_ids
        ]

        # * print after filtering
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")
        # * print the size of different conversation_type
        for conversation_type in self.conversation_types:
            print(f"Number of {conversation_type}: {len([data for data in self.list_data_dict if data.get('conversation_type', 'simple_description') == conversation_type])}")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[:self.data_args.data_debug_num]
            # * print all the scan_id in debug mode, not using for loop
            print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))
        elif self.data_args is not None and self.data_args.split_train_val:
            # * split train and val with 9:1 ratios
            if self.split == 'train':
                self.list_data_dict = self.list_data_dict[:int(self.data_args.split_ratio * len(self.list_data_dict))]
                print(f"Train set size: {len(self.list_data_dict)}")
            else:
                self.list_data_dict = self.list_data_dict[int(self.data_args.split_ratio * len(self.list_data_dict)):]
                print(f"Val set size: {len(self.list_data_dict)}")

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        if not self.use_color:
            point_cloud = point_cloud[:, :3]

        return point_cloud

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        # print(len(self.list_data_dict), sources)
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        if self.point_indicator in sources[0]['conversations'][0]['value']:

            object_id = self.list_data_dict[index]['object_id']

            # Point cloud representation
            point_cloud = self._load_point_cloud(object_id) # * N, C
            if self.normalize_pc:
                point_cloud = self.pc_norm(point_cloud) # * need to norm since point encoder is norm

            if self.tokenizer is None:
                data_dict = dict(
                    pcs=torch.from_numpy(point_cloud.astype(np.float32)),
                    object_ids=object_id
                )
                return data_dict

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.point_backbone_config, point_indicator=self.point_indicator)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_v1(
            sources,
            self.tokenizer)

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # point exist in the data
        if self.point_indicator in self.list_data_dict[index]['conversations'][0]['value']:
            data_dict['pcs'] = torch.from_numpy(point_cloud.astype(np.float32))

        return data_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)




# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

    # def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    #     input_ids, labels = tuple([instance[key] for instance in instances]
    #                               for key in ("input_ids", "labels"))
    #     input_ids = torch.nn.utils.rnn.pad_sequence(
    #         input_ids,
    #         batch_first=True,
    #         padding_value=self.tokenizer.pad_token_id)
    #     labels = torch.nn.utils.rnn.pad_sequence(labels,
    #                                              batch_first=True,
    #                                              padding_value=IGNORE_INDEX)
    #     batch = dict(
    #         input_ids=input_ids,
    #         labels=labels,
    #         attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    #     )

    #     if 'pcs' in instances[0]:
    #         images = [instance['pcs'] for instance in instances]
    #         if all(x is not None and x.shape == images[0].shape for x in images):
    #             batch['pcs'] = torch.stack(images)
    #         else:
    #             batch['pcs'] = images

    #     return batch
    # def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    #     input_ids, labels = tuple([instance[key] for instance in instances]
    #                               for key in ("input_ids", "labels"))
    #     input_ids = torch.nn.utils.rnn.pad_sequence(
    #         input_ids,
    #         batch_first=True,
    #         padding_value=self.tokenizer.pad_token_id)
    #     labels = torch.nn.utils.rnn.pad_sequence(labels,
    #                                              batch_first=True,
    #                                              padding_value=IGNORE_INDEX)
    #     input_ids = input_ids[:, :self.tokenizer.model_max_length]
    #     labels = labels[:, :self.tokenizer.model_max_length]
    #     batch = dict(
    #         input_ids=input_ids,
    #         labels=labels,
    #         attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    #     )

    #     if 'pc' in instances[0]:
    #         pcs = [instance['pc'] for instance in instances]
    #         if all(x is not None and x.shape == pcs[0].shape for x in pcs):
    #             batch['pcs'] = torch.stack(pcs)
    #         else:
    #             batch['pcs'] = pcs

    #     return batch


@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'pcs' in instances[0]:
            point_clouds = [instance['pcs'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['pcs'] = torch.stack(point_clouds)
            else:
                batch['pcs'] = point_clouds # * return as lists

        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset_cls = (LazySupervisedDataset
    #                if data_args.lazy_preprocess else SupervisedDataset)
    # train_dataset = dataset_cls(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             multimodal_cfg=data_args)
    train_dataset = LazySupervisedDataset(
        split='train',
        data_path=data_args.data_path,
        anno_path=data_args.anno_path,
        pointnum=data_args.pointnum,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        use_color=data_args.use_color,
        data_args=data_args
    )
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # if model_args.vision_tower is not None:
    #     if 'mpt' in model_args.model_name_or_path:
    #         model = LlavaMPTForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #         )
    #     else:
    #         model = LlavaLlamaForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=training_args.cache_dir,
    #         )
    # else:
    #     model = transformers.LlamaForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #     )
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    model.config.use_cache = False

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)
    model.get_model().vision_tower.requires_grad_(False)
    if training_args.training_stage == 1:
        # * This will fix all the parameters
        print("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)
        # model.get_model().fix_llm = True
        model.get_model().mm_projector.requires_grad_(True) 
        # model.get_model().vision_tower.requires_grad_(False)
        # model.get_model().vision_tower.requires_grad_(True) # * set as True for fsdp, use fix_pointnet flag to control
        # model.get_model().vision_tower.requires_grad_(False) # * set as True for fsdp, use fix_pointnet flag to control
    else:
        # model.get_model().fix_llm = False
        print("LLM is trainable. Fix_llm flag is set to False")


    # if 'mpt' in model_args.model_name_or_path:
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right"
    #     )
    # else:
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right",
    #         use_fast=False,
    #     )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # import pdb
    # pdb.set_trace()

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if "mpt" in model_args.model_name_or_path:
            conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    # TODO(coco): what's this for? Maybe the `vision_tower` init in LlavaModel?
    # if model_args.vision_tower is not None:
    #     print('=====version: '+ model_args.version)

    #     model_vision_dict = model.get_model().initialize_vision_modules(
    #         vision_tower=model_args.vision_tower,
    #         # mm_vision_select_layer=model_args.mm_vision_select_layer,
    #         pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    #     )
    #     dtype = torch.float32
    #     if training_args.fp16:
    #         dtype = torch.float16
    #     if training_args.bf16:
    #         dtype = torch.bfloat16
    #     print(dtype)
    #     model.get_model().vision_tower.to(dtype=dtype, device=training_args.device)
    #     for layer in model.get_model().vision_tower.modules():
    #         if isinstance(layer, nn.BatchNorm1d):
    #             layer.float()

        # vision_config = model_vision_dict['vision_config']

        # data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_token_len = 1 # pc token is only one token
        # data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        if model_args.tune_mm_mlp_adapter:
            # model.requires_grad_(False)
            print("mm_projection layer is trainable.")
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            model.get_model().mm_projector.requires_grad_(True)
        else:
            model.get_model().mm_projector.requires_grad_(False)
            print("mm_prejcetion layer is fixed.")

            # print('unfreeze vision_tower !!!')
            # for p in model.get_model().vision_tower.parameters():
            #     p.requires_grad = True

        # model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        # if training_args.freeze_mm_mlp_adapter:
        #     for p in model.get_model().mm_projector.parameters():
        #         p.requires_grad = False
        # import pdb
        # pdb.set_trace()
        # if training_args.training_stage == 1:  # stage 1
        #     logging.warning("training stage 1")
            # model.get_model().load_point_backbone_checkpoint(model_args.vision_tower)
        # TODO(coco): dont know what this is for yet
        #     model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
        # else:
        #     # * stage2
        #     model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        # vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        # model.config.sep_image_conv_front = data_args.sep_image_conv_front
        # model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
        #                                   tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)
        if training_args.training_stage == 1:
            # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
            print(f"Default point_backbone_ckpt is {model_args.vision_tower}.")
            model.get_model().load_point_backbone_checkpoint(model_args.vision_tower)
            model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, training_stage=training_args.training_stage)
        else:
            # * stage2
            model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 


        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]

        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)


    data_args.point_backbone_config = model.get_model().point_backbone_config
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    base_lr = training_args.learning_rate
    vision_tower_lr_multiplier = 0.01
    # 获取模型的参数

    # parameters = model.named_parameters()

    # optimizer_grouped_parameters = []
    # for name, param in parameters:
    #     if "vision_tower" in name:
    #         # print(name)
    #         # 如果参数的名称包含"vision_tower"，则将其学习率设置为基础学习率的0.01倍
    #         optimizer_grouped_parameters.append({"params": param, "lr": base_lr * vision_tower_lr_multiplier})
    #     else:
    #         # 其他参数保留基础学习率
    #         optimizer_grouped_parameters.append({"params": param, "lr": base_lr})
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # 定义训练参数
    num_train_steps = int(len(data_module["train_dataset"]) / (training_args.per_device_train_batch_size * training_args.num_gpus * training_args.gradient_accumulation_steps) * training_args.num_train_epochs)
    # warmup_ratio = training_args.warmup_ratio
    print(num_train_steps)
    # 创建学习率调度器对象
    # scheduler = transformers.get_scheduler(
    #     "cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=int(num_train_steps * warmup_ratio),
    #     num_training_steps=num_train_steps
    # )

    # print(scheduler)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    # optimizers=(optimizer, scheduler),
                    **data_module)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # import pdb
    # pdb.set_trace()
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
