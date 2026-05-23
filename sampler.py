#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import datetime
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
from natsort import natsorted

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

from utils import util_net
from utils import util_common
from utils.utility import checkpoint_FiveKNew
from datapipe.datasets import create_dataset


class BaseSampler:
    def __init__(
        self,
        configs,
        use_amp=True,
        padding_offset=16,
        seed=10000,
    ):
        self.logger = checkpoint_FiveKNew(configs)
        self.configs = configs
        self.seed = seed
        self.use_amp = use_amp
        self.padding_offset = padding_offset

        # 固定只支持 [-1, 1]
        self.im_min = -1.0

        self.setup_dist()
        self.setup_seed()
        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        self.num_gpus = 1
        self.rank = 0
        self.device = self.configs.device
        print(self.device)

    def build_model(self):
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)

        model = util_common.instantiate_from_config(self.configs.model).cuda(self.device).half()
        ckpt_path = self.configs.model.ckpt_path
        assert ckpt_path is not None
        print(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.model = model.eval()

        if 'autoencoder' in self.configs:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            print(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda(self.device)
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=self.device)
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)


class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, kwargs=None):
        if noise_repeat:
            self.setup_seed()

        model_kwargs = {} if kwargs is None else kwargs.copy()

        results = self.base_diffusion.p_sample_loop(
            y=y0,
            model=self.model,
            first_stage_model=self.autoencoder,
            noise=None,
            clip_denoised=(self.autoencoder is None),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
        )
        torch.cuda.empty_cache()
        return results

    def inference(self, in_path=None, out_path=None, bs=1, noise_repeat=False, kwargs=None):
        def _process_per_image(im_lq_tensor, kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            autocast_ctx = torch.amp.autocast('cuda') if self.use_amp else nullcontext()

            with autocast_ctx:
                _, _, h, w = im_lq_tensor.shape
                level = len(self.configs.model.params.channel_mult)
                pad_basic = 2 ** (level - 1) * 8
                pad_h = (pad_basic - h % pad_basic) % pad_basic
                pad_w = (pad_basic - w % pad_basic) % pad_basic

                pad_left = pad_right = pad_top = pad_bottom = 0
                if pad_h > 0 or pad_w > 0:
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    im_lq_tensor = F.pad(
                        im_lq_tensor,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='reflect'
                    )

                pred = self.sample_func(im_lq_tensor, noise_repeat=noise_repeat, kwargs=kwargs)
                im_sr_tensor = pred.clamp_(-1.0, 1.0)
                im_sr_tensor = im_sr_tensor[..., pad_top:pad_top + h, pad_left:pad_left + w]

            # 输出保存时转到 [0,1]
            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return {"raw_pred": im_sr_tensor}

        if in_path is None:
            data_config = {
                'type': self.configs.data.test.type,
                'params': self.configs.data.test.params,
            }
            dataset = create_dataset(data_config)
            print(f'Find {len(dataset)} images in test dataset')

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                num_workers=8,
            )

            for data in dataloader:
                micro_data = {key: value for key, value in data.items()}
                im_id = micro_data['image_id'][0]

                kwargs = {
                    key: value.cuda(self.device) if torch.is_tensor(value) else value
                    for key, value in data.items()
                }
                if 'gt' in kwargs:
                    del kwargs['gt']

                preds = _process_per_image(micro_data['rgb'].cuda(self.device), kwargs=kwargs)

                pred = preds['raw_pred'].squeeze(0).cpu().float()
                gt = (micro_data['gt'] * 0.5 + 0.5).squeeze(0).cpu().float()

                assert pred.shape == gt.shape, f'pred shape:{pred.shape} != gt shape:{gt.shape}'

                error_map = torch.abs(gt - pred) * 3
                error_map = torch.clip(error_map, 0, 1)

                psnr = compare_psnr(np.clip(gt.numpy(), 0, 1), np.clip(pred.numpy(), 0, 1))
                mse = compare_mse(np.clip(gt.numpy(), 0, 1), np.clip(pred.numpy(), 0, 1))
                ssim = compare_ssim(np.clip(gt.numpy(), 0, 1), np.clip(pred.numpy(), 0, 1), channel_axis=0, data_range=1)

                self.logger.write_result(psnr, ssim, im_id, micro_data['label'].item())
                self.logger.save_image(pred, im_id, 'pred')
                self.logger.save_image(error_map, im_id, 'error_map')

            self.logger.cal_final_result()

        else:
            in_path = Path(in_path)
            out_path = Path(f'results/{datetime.datetime.now().strftime("%m-%d")}-{self.configs.task}')
            out_view = out_path / 'view'
            out_npz = out_path / 'npz'
            out_view.mkdir(parents=True, exist_ok=True)
            out_npz.mkdir(parents=True, exist_ok=True)

            if in_path.is_dir():
                rgb_paths = natsorted(
                    list(in_path.glob('*.jpg')) +
                    list(in_path.glob('*.png')) +
                    list(in_path.glob('*.jpeg')) +
                    list(in_path.glob('*.bmp'))
                )

                print(f'Find {len(rgb_paths)} images in {in_path}')

                for rgb_path in rgb_paths:
                    image_name = rgb_path.stem
                    pred_view_path = out_view / f'{image_name}.jpg'
                    if pred_view_path.exists():
                        continue

                    rgb_img = Image.open(rgb_path).convert('RGB')
                    rgb_tensor = ToTensor()(rgb_img)
                    rgb_tensor = rgb_tensor * 2 - 1
                    rgb_tensor = rgb_tensor.unsqueeze(0).cuda(self.device)

                    kwargs = {
                        'bayer_pattern': torch.tensor([0], device=self.device),
                        'label': torch.tensor([0], device=self.device),
                    }

                    preds = _process_per_image(rgb_tensor, kwargs=kwargs)
                    pred = preds['raw_pred'].squeeze(0).cpu().float()

                    pred_view = torch.stack([
                        pred[0],
                        (pred[1] + pred[2]) / 2,
                        pred[3],
                    ], dim=0) if pred.shape[0] == 4 else pred

                    pred_view = (pred_view.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                    Image.fromarray(pred_view).save(pred_view_path)

                    pred_save = pred.permute(1, 2, 0).numpy()
                    pred_save = (pred_save * 65535).astype(np.uint16)
                    np.savez_compressed(out_npz / f'{image_name}.npz', raw=pred_save, wl=65535, bl=0)

                    print(f"{image_name} saved to {pred_view_path}")

            else:
                if not in_path.exists():
                    raise FileNotFoundError(f"Input file {in_path} not found")

                rgb_img = Image.open(in_path).convert('RGB')
                rgb_tensor = ToTensor()(rgb_img)
                rgb_tensor = rgb_tensor * 2 - 1
                rgb_tensor = rgb_tensor.unsqueeze(0).cuda(self.device)

                kwargs = {
                    'bayer_pattern': torch.tensor([0], device=self.device),
                    'label': torch.tensor([0], device=self.device),
                }

                preds = _process_per_image(rgb_tensor, kwargs=kwargs)
                pred = preds['raw_pred'].squeeze(0).cpu().float()

                pred_view = torch.stack([
                    pred[0],
                    (pred[1] + pred[2]) / 2,
                    pred[3],
                ], dim=0) if pred.shape[0] == 4 else pred

                pred_view = (pred_view.permute(1, 2, 0) * 255).numpy().astype(np.uint8)

                image_name = in_path.stem
                pred_view_path = out_view / f'{image_name}.jpg'
                Image.fromarray(pred_view).save(pred_view_path)

                pred_save = pred.permute(1, 2, 0).numpy()
                pred_save = (pred_save * 65535).astype(np.uint16)
                np.savez_compressed(out_npz / f'{image_name}.npz', raw=pred_save, wl=65535, bl=0)

                print(f"{image_name} saved to {pred_view_path}")


if __name__ == '__main__':
    pass