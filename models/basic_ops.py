"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn

def bright_mask_loss(pred, target, mask, min_weight=1.0, max_weight=5.0, balance_factor=1.0):
    assert pred.shape == target.shape
    assert mask.shape[2:] == pred.shape[2:]
    assert mask.shape[0] == pred.shape[0]

    if mask.shape[0] == 1:
        mask = mask.repeat(pred.size(0), 1, 1, 1)
    
    # 计算基础像素级别的MSE
    pixel_loss = (pred - target) ** 2  # [B, C, H, W]
    # 分别计算掩码区域和非掩码区域的平均损失
    mask_loss = (pixel_loss * mask).sum(dim=(2,3)) / (mask.sum(dim=(2,3)) + 1e-6)
    non_mask_loss = (pixel_loss * (1 - mask)).sum(dim=(2,3)) / ((1 - mask).sum(dim=(2,3)) + 1e-6)

    # 根据mask区域loss的大小来调整权重
    loss_ratio = mask_loss / (non_mask_loss + 1e-6) / 10
    adaptive_weight = min_weight + (max_weight - min_weight) * th.sigmoid(balance_factor * (loss_ratio - 1.0))
    adaptive_weight = adaptive_weight[..., None, None]
    
    # 创建权重图
    weight_map = mask * (adaptive_weight - 1.0) + 1.0
    
    # 应用权重并计算最终损失
    weighted_loss = pixel_loss * weight_map
    
    return weighted_loss

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

def normalization12(channels):
    """
    Make a standard normalization layer.
    """
    return GroupNorm12(12, channels)

def normalization_any(channels, group_num):
    return GroupNorm12(group_num, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class GroupNorm12(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
        
if __name__ == '__main__':
    a = th.ones(2,3,5,5)
    mask = th.tensor([1,2,3]).reshape(1,3,1,1)
    a*=mask
    b = mean_flat(a)
    print(a)
    print(b)
    '''
    out:    tensor([2., 2.])
    对一个batch求mse
    '''

