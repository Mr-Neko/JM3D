import argparse
import torch
import torch.utils.data as data

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.train.train import farthest_point_sample, random_point_dropout, random_scale_point_cloud, shift_point_cloud, rotate_perturbation_point_cloud, rotate_point_cloud, pc_normalize

from PIL import Image

import requests
from PIL import Image
# from io import BytesIO
from llava.train.io import IO
import numpy as np
import os
import math
import types, pickle, plyfile
import tqdm

class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        self.sets = ''# 'Hard'

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print('Processing data %s (only running in the first time)...' % self.save_path)
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print('Load processed data from %s...' % self.save_path)
                    print('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...'.format(self.npoints))
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

        self.hard_set = ['cone', 'curtain', 'door', 'dresser', 'glass_box', 'mantel', 'night_stand', 'person', 'plant', 'radio', 'range_hood', 'sink', 'stairs', 'tent', 'toilet', 'tv_stand', 'xbox']
        self.medium_set = ['cone', 'cup', 'curtain', 'door', 'dresser', 'glass_box', 'mantel', 'monitor', 'night_stand', 'person', 'plant', 'radio', 'range_hood', 'sink', 'stairs', 'stool', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        if self.sets == 'Medium':
            if not label_name in self.medium_set:
                print(index, label_name, 'not in medium set')
                return None, None, None
            label = self.medium_set.index(label_name)
            
        elif self.sets == 'Hard':
            if not label_name in self.hard_set:
                print(index, label_name, 'not in hard set')
                return None, None, None
            label = self.hard_set.index(label_name)
        return current_points, label, label_name

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def pc_processor(pc_file, pc_dataset):
    if pc_dataset == 'shapenet' or pc_dataset == 'objaverse':
        # if pc_file.endswith('.pt'):
        #     pc_data = torch.load(os.path.join(pc_file))
        #     pc_data = pc_data[:3,:].permute(1, 0).numpy().astype(np.float32)
        # else:
        #     pc_data = IO.get(os.path.join(pc_file)).astype(np.float32)

        # pc_data = np.load(pc_file)['arr_0']
        pc_data = np.load(pc_file)
        # if pc_file.endswith('.pt'):
        #     # pc_data = torch.load(os.path.join(self.pc_folder, pc_file))
        #     pc_data_name = f'{pc_file[:-3]}_8192'
        #     try:
        #         pc_data = np.load(os.path.join(self.pc_folder, pc_file[:-3], pc_data_name + '.npz'))
        #     except:
        #         # pc_data = np.load(os.path.join(self.pc_folder, pc_file[:-3], pc_data_name + '.npy'))
        #         assert('fail to load npz file !!!!!!!!!!!')

        #     # pc_data = pc_data[:3,:].permute(1, 0).numpy().astype(np.float32)
        #     pc_data = pc_data['arr_0']
        # else:
        #     pc_data = IO.get(os.path.join(pc_file)).astype(np.float32)


        # print(pc_data.shape)
        uniform = False
        augment = False
        use_height = False
        npoints = 8192
        permutation = np.arange(npoints)
        def random_sample(pc, num):
            np.random.shuffle(permutation)
            pc = pc[permutation[:num]]
            return pc
        def pc_norm(pc):
            """ pc: NxC, return NxC """
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc = pc / m
            return pc
        if uniform and npoints < pc_data.shape[0]:
            pc_data = farthest_point_sample(pc_data, npoints)
        else:
            pc_data = random_sample(pc_data, npoints)
        pc_data = pc_norm(pc_data)
        # print(pc_data.shape)

        if augment:
            pc_data = random_point_dropout(pc_data[None, ...])
            pc_data = random_scale_point_cloud(pc_data)
            pc_data = shift_point_cloud(pc_data)
            pc_data = rotate_perturbation_point_cloud(pc_data)
            pc_data = rotate_point_cloud(pc_data)
            pc_data = pc_data.squeeze()

        if use_height:
            gravity_dim = 1
            height_array = pc_data[:, gravity_dim:gravity_dim + 1] - pc_data[:,
                                                                    gravity_dim:gravity_dim + 1].min()
            pc_data = np.concatenate((pc_data, height_array), axis=1)
            pc_data = torch.from_numpy(pc_data).float()
        else:
            pc_data = torch.from_numpy(pc_data).float()

        return pc_data
    elif pc_dataset =='modelnet':
        dataset = ModelNet(types.SimpleNamespace(**{
            'DATA_PATH': pc_file,
            'npoints': 8192,
            'USE_NORMALS': False,
            'NUM_CATEGORY': 40,
            'subset': 'test',
            'use_height': False
        }))
        random_idx = np.random.randint(0, len(dataset))
        pc_data, label, label_name = dataset[random_idx]
        pc_data = pc_data.numpy()
        pc_data = torch.from_numpy(pc_data).float()
        print('ModelNet test sample label:', label_name, '| shape:', pc_data.shape, '| max & min:', pc_data.max(), pc_data.min())
        return pc_data
    elif pc_dataset == 'scannet':
        f = plyfile.PlyData().read(pc_file)
        points = np.array([list(x) for x in f.elements[0]])
        coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
        coords = pc_normalize(coords)
        colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
        print('Scannet sample shape:', coords.shape, ' | max & min:', coords.max(), coords.min())
        return torch.from_numpy(coords).float()
    # elif pc_dataset == 'objaverse':

    else:
        raise ValueError(f'Unknown pc_dataset: {pc_dataset}')
    # print(pc_data.shape, pc_data)
    

# def load_image(image_file):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     return image


def eval_model(args):
    # Model
    disable_torch_init()

    # model_name = get_model_name_from_path(args.model_path)
    model_name = args.model_name
    print('model_name', model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, torch_dtype=args.torch_dtype)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)    
    model.eval()

    import pdb
    pdb.set_trace()
    # vision_tower = model.get_vision_tower()
    # vision_backbone = torch.load('backbones/pointmlp/pointmlp_backbone.pt', map_location='cpu')
    # vision_tower.load_state_dict(vision_backbone['state_dict'], strict=True)
    # for layer in vision_tower.modules():
    #     if isinstance(layer, torch.nn.BatchNorm1d):
    #         layer.float()
    
    # vision_tower = model.get_model().vision_tower
    # for name, param in vision_tower.named_parameters():
    #     print(name, param.dtype)

    # for layer in vision_tower.modules():
    #     if isinstance(layer, torch.nn.BatchNorm1d):
    #         print(layer)
    #         layer.eval()

    qs = args.query
    if model.config.mm_use_im_start_end:
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_PATCH_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    print(args.conv_mode)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    while 1:
        # pc_file = './data/Objaverse/Cap3D_pcs_pt/' + input('Enter point cloud id: ') + '.pt'
        pc_file = args.pc_file
        if pc_file == 'exit':
            break
        pc_tensor = pc_processor(pc_file, args.pc_dataset)
        # pc_tensor = pc_processor(args.pc_file, args.pc_dataset)
        # print(pc_tensor.shape)
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # print(input_ids.shape)
        # exit()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # import pdb
        # pdb.set_trace()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # pcs=pc_tensor.unsqueeze(0).half().cuda(),
                pcs=pc_tensor.unsqueeze(0).cuda(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         pcs=pc_tensor.unsqueeze_(0).cuda(),
        #         do_sample=True,
        #         temperature=0.2,
        #         top_k=50,
        #         max_length=2048,
        #         top_p=0.95,
        #         stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print('Question:', qs)
        print('Point Cloud:', args.pc_file)
        print('Answer:', outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default="llava")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--pc_file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='v1')
    parser.add_argument("--pc_dataset", type=str, default='shapenet', help='shapenet / modelnet / scannet')
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]


    eval_model(args)
