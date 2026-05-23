#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import math
import time
import random
import datetime
import functools
import shutil
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from einops import rearrange

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from datapipe.datasets import create_dataset
from utils import util_net
from utils import util_common
from utils import util_image


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        self.setup_dist()
        self.setup_seed()

        self.device = self.configs.device

        self.weight_l2 = self.configs.trainer.get('weight_l2', 1.0)
        self.weight_l1 = self.configs.trainer.get('weight_l1', 0.0)
        self.weight_logl1 = self.configs.trainer.get('weight_logl1', 0.0)

        print(self.device)
        print('weight_l2:', self.weight_l2, 'weight_l1:', self.weight_l1, 'weight_logl1:', self.weight_logl1)

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://',
            )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0
        print('rank:', self.rank)
        print('gpu_num:', self.num_gpus)

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)

        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            if self.configs.tasks != 'None':
                project_id = project_id + '-' + self.configs.tasks
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        logtext_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtext_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtext_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        if self.rank == 0:
            task_path = save_dir / 'task.txt'
            task_path.write_text("")

        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and not ckpt_dir.exists():
            ckpt_dir.mkdir()

        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float)
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and not ema_ckpt_dir.exists():
                ema_ckpt_dir.mkdir()

        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))
            try:
                shutil.copy(self.configs.cfg_path, str(save_dir / 'config.yaml'))
            except Exception:
                print("config file exist.")

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt:
                    ema_state[key] = deepcopy(ckpt['module.' + key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")

            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start + 1):
                self.adjust_lr(ii)

            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_" + Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)

            torch.cuda.empty_cache()

            if self.amp_scaler is not None and "amp_scaler" in ckpt:
                self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                if self.rank == 0:
                    self.logger.info("Loading scaler from resumed state...")

            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.configs.train.lr,
            weight_decay=self.configs.train.weight_decay,
        )
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()

        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)

        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")

        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank], static_graph=False, find_unused_parameters=False)
        else:
            self.model = model

        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key: deepcopy(value.data) for key, value in self.model.state_dict().items()}
            )
            self.ema_ignore_keys = [
                x for x in self.ema_state.keys()
                if ('running_' in x or 'num_batches_tracked' in x)
            ]

        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True:
                yield from loader

        datasets = {'train': create_dataset(self.configs.data.get('train', dict))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))

        if self.rank == 0:
            for phase in datasets.keys():
                self.logger.info(f'Number of images in {phase} data set: {len(datasets[phase])}')

        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=self.num_gpus,
                rank=self.rank,
            )
        else:
            sampler = None

        train_num_workers = min(self.configs.train.num_workers, 8)
        train_loader_kwargs = dict(
            dataset=datasets['train'],
            batch_size=self.configs.train.batch[0] // max(self.num_gpus, 1),
            shuffle=False if self.num_gpus > 1 else True,
            drop_last=True,
            num_workers=train_num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        if train_num_workers > 0:
            train_loader_kwargs['prefetch_factor'] = self.configs.train.get('prefetch_factor', 2)
            train_loader_kwargs['worker_init_fn'] = my_worker_init_fn

        dataloaders = {
            'train': _wrap_loader(udata.DataLoader(**train_loader_kwargs))
        }

        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(
                datasets['val'],
                batch_size=self.configs.train.batch[1],
                shuffle=False,
                drop_last=True,
                num_workers=0,
                pin_memory=True,
                sampler=None,
            )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32):
        out = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                value = value.cuda(non_blocking=True)
                if torch.is_floating_point(value):
                    value = value.to(dtype=dtype)
            out[key] = value
        return out

    def validation(self):
        raise NotImplementedError

    def train(self):
        self.init_logger()
        self.build_model()
        self.setup_optimizaton()
        self.resume_from_ckpt()
        self.build_dataloader()

        self.model.train()

        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            data = self.prepare_data(next(self.dataloaders['train']))
            self.training_step(data, iter=ii)

            self.adjust_lr()

            if (ii + 1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if 'val' in self.dataloaders and (ii + 1) % self.configs.train.get('val_freq', 10000) == 0:
                torch.cuda.empty_cache()
                self.validation()
                torch.cuda.empty_cache()

            if (ii + 1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii + 1)

        self.close_logger()

    def training_step(self, data, iter=0):
        raise NotImplementedError

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / f'model_{self.current_iters}.pth'
            ckpt = {
                'iters_start': self.current_iters,
                'log_step': {phase: self.log_step[phase] for phase in ['train', 'val']},
                'log_step_img': {phase: self.log_step_img[phase] for phase in ['train', 'val']},
                'state_dict': self.model.state_dict(),
            }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)

            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / f'ema_model_{self.current_iters}.pth'
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]: value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1 - rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        assert self.tf_logging or self.local_logging

        if im_tensor.shape[1] == 4:
            im_tensor = torch.stack(
                (im_tensor[:, 0, :, :], im_tensor[:, 1, :, :], im_tensor[:, -1, :, :]),
                dim=1,
            )
        elif im_tensor.shape[1] == 12:
            im_tensor = nn.PixelShuffle(2)(im_tensor)

        im_tensor = vutils.make_grid(
            im_tensor,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1),
            scale_each=False,
        )
        im_tensor = im_tensor.clamp_(-1.0, 1.0)

        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.jpg")
            im_np = im_tensor.cpu().permute(1, 2, 0).numpy()
            util_image.imwrite(im_np, im_path)

        if self.tf_logging:
            self.writer.add_image(
                f"{phase}-{tag}-{self.log_step_img[phase]}",
                im_tensor,
                self.log_step_img[phase],
            )

        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1

    def load_model(self, model, ckpt_path=None):
        if self.rank == 0:
            self.logger.info(f'Loading from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            self.logger.info('Loaded Done')


class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            warmup_iters = self.configs.train.warmup_iterations
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.configs.train.iterations - warmup_iters,
                eta_min=self.configs.train.lr_min,
            )

    def build_model(self):
        super().build_model()

        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend(
                [x for x in self.ema_state.keys() if 'relative_position_index' in x]
            )

        if 'autoencoder' in self.configs:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")

            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.to(self.device)

            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            autoencoder.load_state_dict(ckpt, strict=True)

            for p in autoencoder.parameters():
                p.requires_grad_(False)
            autoencoder.eval()

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")

            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    def backward_step(self, dif_loss_wrapper, num_grad_accumulate):
        context = torch.amp.autocast('cuda') if self.configs.train.use_amp else nullcontext()
        with context:
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = 0
            for key, value in losses.items():
                if key != 'loss':
                    losses['loss'] += value
            loss = losses['loss'].mean() / num_grad_accumulate

        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data, iter=0):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        log_loss = {}
        self.optimizer.zero_grad(set_to_none=True)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key: value[jj:jj + micro_batchsize] if torch.is_tensor(value) else value for key, value in data.items()}
            last_batch = (jj + micro_batchsize >= current_batchsize)

            tt = torch.randint(
                0,
                self.base_diffusion.num_timesteps,
                size=(micro_data['gt'].shape[0],),
                device=f"cuda:{self.rank}",
            )

            model_kwargs = {}

            for key, value in micro_data.items():
                model_kwargs[key] = value

            if 'label' in micro_data:
                model_kwargs['label'] = micro_data['label'].int()

            model_kwargs['iter'] = iter

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                model=self.model,
                x_start=micro_data['gt'],
                y=micro_data['rgb'],
                t=tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=None,
                weight_l2=self.weight_l2,
                weight_l1=self.weight_l1,
                weight_logl1=self.weight_logl1,
            )

            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, num_grad_accumulate)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, num_grad_accumulate)

            for key, value in losses.items():
                if key in log_loss:
                    log_loss[key] += value.detach()
                else:
                    log_loss[key] = value.detach()

            if last_batch:
                if self.amp_scaler is not None:
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if hasattr(self.configs.train, 'ema_rate'):
                    self.update_ema_model()

                self.log_step_train(log_loss, tt, micro_data, z_t, z0_pred.detach())

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters

        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        if self.rank != 0:
            return

        num_timesteps = self.base_diffusion.num_timesteps
        record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]

        if self.current_iters % self.configs.train.log_freq[0] == 1:
            self.loss_mean = {
                key: torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                for key in loss.keys()
            }
            self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)

        for key in loss.keys():
            if key not in self.loss_mean:
                self.loss_mean[key] = torch.zeros(size=(len(record_steps),), dtype=torch.float64)

        for jj in range(len(record_steps)):
            index = record_steps[jj] - 1
            mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
            self.loss_count[jj] += mask.sum().item()

            for key, value in loss.items():
                assert value.shape == mask.shape
                current_loss = torch.sum(value.detach() * mask)
                self.loss_mean[key][jj] += current_loss.item()

        if self.current_iters % self.configs.train.log_freq[0] == 0:
            if torch.any(self.loss_count == 0):
                self.loss_count += 1e-4

            for key in loss.keys():
                self.loss_mean[key] /= self.loss_count

            loss_names = '/'.join(loss.keys())
            log_str = f'Train: {self.current_iters:06d}/{self.configs.train.iterations:06d}, {loss_names}: '

            format_str = 't({:d}):' + '{:.2e}/'.join([''] * (len(loss.keys()) + 1))
            for jj, current_record in enumerate(record_steps):
                values = [current_record]
                for key in loss.keys():
                    values.append(self.loss_mean[key][jj].item())
                log_str += format_str.format(*values) + ', '

            log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
            self.logger.info(log_str)
            self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)

        if self.current_iters % self.configs.train.log_freq[1] == 0:
            x0_pred = self.base_diffusion.decode_first_stage(z0_pred, self.autoencoder)
            x_t = self.base_diffusion.decode_first_stage(
                self.base_diffusion._scale_input(z_t, tt),
                self.autoencoder,
            )

            self.logging_image(batch['rgb'], tag=f'lq_{self.current_iters}', phase=phase, add_global_step=False)
            self.logging_image(batch['gt'], tag=f'gt_{self.current_iters}', phase=phase, add_global_step=False)
            self.logging_image(x_t, tag=f'diffused_{self.current_iters}', phase=phase, add_global_step=False)
            self.logging_image(x0_pred, tag=f'x0-pred_{self.current_iters}', phase=phase, add_global_step=True)

        if self.current_iters % self.configs.train.save_freq == 1:
            self.tic = time.time()
        if self.current_iters % self.configs.train.save_freq == 0:
            self.toc = time.time()
            elapsed = self.toc - self.tic
            self.logger.info(f"Elapsed time: {elapsed:.2f}s")
            self.logger.info("=" * 100)

    def validation(self, phase='val'):
        if self.rank != 0:
            return

        if self.configs.train.use_ema_val:
            self.reload_ema_model()
            self.ema_model.eval()
        else:
            self.model.eval()

        torch.cuda.empty_cache()

        indices = np.linspace(
            0,
            self.base_diffusion.num_timesteps,
            self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
            endpoint=False,
            dtype=np.int64,
        ).tolist()
        if (self.base_diffusion.num_timesteps - 1) not in indices:
            indices.append(self.base_diffusion.num_timesteps - 1)

        batch_size = self.configs.train.batch[1]
        num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)

        mean_psnr = 0

        for ii, data in enumerate(self.dataloaders[phase]):
            if ii + 1 > num_iters_epoch:
                break

            data = self.prepare_data(data)

            _, _, h, w = data['rgb'].shape
            level = len(self.configs.model.params.channel_mult)
            pad_basic = 2 ** (level - 1) * 8
            pad_h = (pad_basic - h % pad_basic) % pad_basic
            pad_w = (pad_basic - w % pad_basic) % pad_basic

            if pad_h > 0 or pad_w > 0:
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                data['rgb'] = F.pad(
                    data['rgb'],
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='reflect',
                )
            else:
                pad_left = pad_right = pad_top = pad_bottom = 0

            if 'gt' in data:
                im_lq, im_gt = data['rgb'], data['gt']
            else:
                im_lq = data['rgb']

            model_kwargs = {'rgb': data['rgb']}
            for key, value in data.items():
                if key != 'gt':
                    model_kwargs[key] = value
            if 'label' in data:
                model_kwargs['label'] = data['label'].int()

            model_kwargs['iter'] = self.current_iters - 1

            num_iters = 0
            with torch.no_grad():
                autocast_ctx = torch.amp.autocast('cuda') if self.configs.train.use_amp else nullcontext()
                with autocast_ctx:
                    for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                    ):
                        sample_decode = {}
                        if num_iters in indices:
                            for key, value in sample.items():
                                if key in ['sample']:
                                    sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        value,
                                        self.autoencoder,
                                    ).clamp(-1.0, 1.0)
                                    sample_decode[key] = sample_decode[key][..., pad_top:pad_top + h, pad_left:pad_left + w]

                            im_sr_progress = sample_decode['sample']
                            if num_iters + 1 == 1:
                                im_sr_all = im_sr_progress
                            else:
                                im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)

                        num_iters += 1

            if 'gt' in data:
                mean_psnr += util_image.batch_PSNR(
                    sample_decode['sample'] * 0.5 + 0.5,
                    im_gt * 0.5 + 0.5,
                    ycbcr=self.configs.train.val_y_channel,
                )

            if (ii + 1) % self.configs.train.log_freq[2] == 0:
                self.logger.info(f'Validation: {ii + 1:02d}/{num_iters_epoch:02d}...')

                ch = 4 if im_lq.shape[1] == 12 else im_lq.shape[1]
                im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=ch)
                self.logging_image(
                    im_sr_all,
                    tag='progress',
                    phase=phase,
                    add_global_step=False,
                    nrow=len(indices),
                )
                if 'gt' in data:
                    self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                self.logging_image(im_lq, tag=f'lq_{self.current_iters}', phase=phase, add_global_step=True)

        if 'gt' in data:
            mean_psnr /= len(self.datasets[phase])
            self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}...')
            self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)

        self.logger.info("=" * 100)
        torch.cuda.empty_cache()

        if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
            self.model.train()


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)