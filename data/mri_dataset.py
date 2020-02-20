import os
import numpy as np
import torch
import json
from data.base_dataset import BaseDataset, __scale_width, __crop, __make_power_2, __flip, get_params
from data.image_folder import make_dataset
from torchvision import transforms
from PIL import Image

class MRI_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.statistical_file = 'statistics.json'
        self.params = {}
        self.tp_params = ['means', 'stds']
        self.get_data_folder_name(self.root, opt.config_data)

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else self.config_data['label_folder']
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else self.config_data['image_folder']
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))
            statistics = get_statistics(os.path.join(self.dir_B, self.statistical_file), 
                                        self.tp_params)
            self.params.update(statistics)
            opt.statistics = statistics

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, 
                            opt.phase + self.config_data['instance_folder'])
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        self.dataset_size = len(self.A_paths)
        

    def __getitem__(self, index):
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        self.get_params(A.size)
        tf_A_list = get_transform_label(self.opt, method=Image.NEAREST, **self.params)
        tf_A = transforms.Compose(tf_A_list)
        A_tensor = tf_A(A)

        B_tensor = inst_tensor = feat_tensor = 0
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        tf_B_list = get_transform_image(tf_A_list, self.opt, normalize=False, **self.params)
        B_tensor = transforms.Compose(tf_B_list)(B)

        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = tf_A(inst)

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def get_params(self, size):
        crop_flip = get_params(self.opt, size)
        self.params.update(crop_flip)    

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'MRI Dataset'

    def get_data_folder_name(self, root_fd, config_data_name):
        data_fd = os.path.split(root_fd)[-2]
        path_cfg_data = os.path.join(data_fd, config_data_name)
        with open(path_cfg_data, 'r') as cfg_data_file:
            config_data = json.load(cfg_data_file)
        self.config_data = config_data

def get_transform_label(opt, method=Image.BICUBIC, **params):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
    
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))
    
    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list.append(transforms.Lambda(lambda img: __toTensor(img)))
    return transform_list

def get_transform_image(tf_list_label, opt, normalize=True, **params):
    transform_list = tf_list_label
    # transform_list.append(transforms.Lambda(lambda tensor: 
    #                             __scale_01_Tensor(tensor, params['min_vox'], params['max_vox'])))
    transform_list.append(transforms.Lambda(lambda tensor: __scale_mean_tensor(tensor,)))
    if normalize:
        transform_list.append(transforms.Lambda(lambda tensor: 
                                __normalize(tensor, params['means'], params['stds'])))
    
    return transform_list

def __toTensor(image):
    arr = np.asarray(image, dtype=np.float32)
    ts = torch.from_numpy(np.expand_dims(arr, 0))
    return ts

def __scale_01_Tensor(tensor, min_voxel, max_voxel):
    dtype = tensor.dtype
    min_ts = torch.as_tensor(min_voxel, dtype=dtype, device=tensor.device)
    max_ts = torch.as_tensor(max_voxel, dtype=dtype, device=tensor.device)
    tensor.div_(max_ts - min_ts)
    return tensor

def __scale_mean_tensor(tensor):
    mean_ts = torch.mean(tensor)
    tensor.div_(mean_ts)
    return tensor

def __scale_min_max_tensor(tensor):
    min_ts = torch.min(tensor)
    max_ts = torch.max(tensor)
    diff = max_ts - min_ts
    return tensor.div_(diff)

def __normalize(tensor, means, stds):
    dtype = tensor.dtype
    mean = torch.as_tensor(means, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(stds, dtype=dtype, device=tensor.device)
    tensor.sub_(mean).div_(std)
    return tensor

def get_statistics(statistical_file, tuple_params):
    with open(statistical_file, 'r') as sts_file:
        statistics = json.load(sts_file)
    
    for k, v in statistics.items():
        if k in tuple_params:
            statistics[k] = eval(v)

    return statistics