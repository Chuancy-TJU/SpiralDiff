import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils import util_image
from utils import util_common
import glob, os
from glob import glob as glob
from PIL import Image as PILImage

import torchvision.transforms as TF
import torch.nn.functional as F

def create_dataset(dataset_config):
    if dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    elif dataset_config['type'] == 'rgb2raw':
        dataset = RGB2RAWDataset(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

def crop(raw, rgb, patch_size):
    C, H, W = raw.shape
    c, h, w = rgb.shape
    if c == 3 and C == 4:
        rgb = F.pixel_unshuffle(rgb, 2)
        h = h / 2
        w = w / 2
    assert H == h and W == w

    start_H = random.randint(0, H - patch_size)
    start_W = random.randint(0, W - patch_size)

    patch_raw = raw[:, start_H:start_H + patch_size, start_W:start_W + patch_size]
    patch_rgb = rgb[:, start_H:start_H + patch_size, start_W:start_W + patch_size]

    if c == 3 and C == 4:
        patch_rgb = F.pixel_shuffle(patch_rgb, 2)
    return patch_raw, patch_rgb

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            extra_dir_path=None,
            extra_transform_type=None,
            extra_transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp', 'npy'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))  # 获取该路径下全部的图片，返回图片名列表
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)  # 取全部还是随机取一些
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)  # 对LR进行缩放？？？？

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            assert extra_transform_type is not None
            self.extra_transform = get_transforms(extra_transform_type, extra_transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='float32')
        # im_base = np.load(im_path_base)

        im_target = self.transform(im_base)  # 为什么要transform呢？
        out = {'image':im_target, 'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base
        
        # out = {'image':im_target, 'lq':im_target, 'gt':im_target}

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class RGB2RAWDataset(Dataset):
    def __init__(
        self,
        mode='train',
        root='/home/xsb/dataset/mix/mix_data',
        camera_name='combined',
        patch_size=256,
    ):
        super().__init__()

        assert mode in ['train', 'val', 'test'], f'Unsupported mode: {mode}'

        self.mode = mode
        self.root = root
        self.patch_size = patch_size
        self.camera_name = camera_name

        self.samples = []

        mode_folder = 'test' if mode == 'val' else mode

        camera_to_path = {
            "Canon EOS 5D": os.path.join(root, f'{mode_folder}/Canon EOS 5D'),
            "NIKON D700": os.path.join(root, f'{mode_folder}/NIKON D700'),
            "Nikon750": os.path.join(root, f'{mode_folder}/Nikon750'),
            "Sony_RX100m7": os.path.join(root, f'{mode_folder}/Sony_RX100m7'),
        }
        ntire_paths = {
            'iPhone-X': os.path.join(root, f'{mode_folder}/iPhone-X'),
            'Samsung-s9': os.path.join(root, f'{mode_folder}/Samsung-s9'),
        }

        if self.camera_name == "combined":
            data_roots = list(camera_to_path.values())
        elif self.camera_name == 'ntire':
            data_roots = list(ntire_paths.values())
        elif self.camera_name in ['iPhone-X', 'Samsung-s9']:
            data_roots = [ntire_paths[self.camera_name]]
        elif self.camera_name in camera_to_path:
            data_roots = [camera_to_path[self.camera_name]]
        else:
            raise ValueError(f'Unsupported camera_name: {self.camera_name}')

        def get_sort_key(path):
            return os.path.splitext(os.path.basename(path))[0]

        for data_root in data_roots:
            if mode == 'train':
                rgb_pattern = os.path.join(data_root, 'RGB', '*', '*.jpg')
                raw_pattern = os.path.join(data_root, 'RAW', '*', '*.npz')
            else:
                rgb_pattern = os.path.join(data_root, 'RGB', '*.jpg')
                raw_pattern = os.path.join(data_root, 'RAW', '*.npz')

            rgb_paths = glob(rgb_pattern)
            raw_paths = glob(raw_pattern)

            if len(rgb_paths) == 0 or len(raw_paths) == 0:
                rgb_paths = glob(os.path.join(data_root, 'RGB', '*.jpg'))
                raw_paths = glob(os.path.join(data_root, 'RAW', '*.npz'))

            rgb_paths = sorted(rgb_paths, key=get_sort_key)
            raw_paths = sorted(raw_paths, key=get_sort_key)

            assert len(rgb_paths) == len(raw_paths), f"Length mismatch in {data_root}"
            for rgb_path, raw_path in zip(rgb_paths, raw_paths):
                rgb_name = os.path.splitext(os.path.basename(rgb_path))[0]
                raw_name = os.path.splitext(os.path.basename(raw_path))[0]
                assert rgb_name == raw_name, f"Mismatch: {rgb_path} vs {raw_path}"

            print(f"{os.path.basename(data_root)}: All files matched correctly.")

            stride = 4 if self.camera_name != 'mix' else 10
            if mode == 'val':
                rgb_paths = rgb_paths[::stride]
                raw_paths = raw_paths[::stride]

            self.samples.extend(zip(rgb_paths, raw_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        out = {}

        rgb_path, raw_path = self.samples[index]

        # RGB: [0,1] -> [-1,1], then pixel_unshuffle to 12-channel
        rgb_img = PILImage.open(rgb_path).convert('RGB')
        rgb_img = TF.ToTensor()(rgb_img)          # 3 x H x W, [0,1]
        rgb_img = rgb_img * 2.0 - 1.0            # [-1,1]
        rgb_img = F.pixel_unshuffle(rgb_img, 2)  # 12 x H/2 x W/2

        # RAW
        raw = np.load(raw_path)
        assert 'raw' in raw, f"'raw' field missing in {raw_path}"
        assert 'bl' in raw, f"'bl' field missing in {raw_path}"
        assert 'wl' in raw, f"'wl' field missing in {raw_path}"

        raw_img = raw['raw'].astype(np.float32)
        bl = raw['bl'].item()
        wl = raw['wl'].item()

        if 'bp' in raw:
            bayer_pattern = str(raw['bp'])
        else:
            bayer_pattern = 'RGGB'

        raw_img = ((raw_img - bl) / (wl - bl)).clip(0, 1)
        raw_img = torch.from_numpy(raw_img).permute(2, 0, 1)  # 4 x H x W
        raw_img = raw_img * 2.0 - 1.0                         # [-1,1]

        if self.mode == 'train':
            raw_img, rgb_img = crop(raw_img, rgb_img, self.patch_size)

        # if self.mode == 'val':
        #     _, h, w = raw_img.shape
        #     if h > 1980 or w > 1980:
        #         raw_img = raw_img[:, :1980, :1980]
        #         rgb_img = rgb_img[:, :1980, :1980]

        disc = {"RGGB": 0, "BGGR": 1, "GRBG": 2, "GBRG": 3}
        assert bayer_pattern in disc, f"Unknown bayer pattern: {bayer_pattern}"
        bayer_pattern_idx = disc[bayer_pattern]

        label = raw['label'] if 'label' in raw else None
        assert label is not None, f"'label' field missing in {raw_path}"

        out['rgb'] = rgb_img
        out['gt'] = raw_img
        out['bayer_pattern'] = bayer_pattern_idx
        out['label'] = label

        if self.mode == 'test':
            out['image_id'] = os.path.splitext(os.path.basename(rgb_path))[0].split('-')[0]

        return out