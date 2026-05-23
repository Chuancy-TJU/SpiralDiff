#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default=None, help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-label", "--label", type=int, default=0, help="Camera_id.")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--task", type=str, default=None, help="Tasks.")
    parser.add_argument("--folder", type=str, default=None, help="Folder path.")
    parser.add_argument(
            "--device",
            type=str,
            default='cuda:0',
            )
    parser.add_argument(
            "--ckpt",
            type=str,
            )
    parser.add_argument(
            "--yaml",
            type=str,
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load(args.yaml)
    ckpt_path = Path(args.ckpt)


    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.sf = args.scale
    configs.device = args.device
    configs.out_path = args.out_path
    configs.ckpt = args.ckpt
    configs.yaml = args.yaml
    configs.task = args.task
    configs.folder = args.folder

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)


    resshift_sampler = ResShiftSampler(
            configs,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )

    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            bs=1,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
