# -----------------------------------------------------------------------------------
# Swin transformer.
# https://arxiv.org/pdf/2103.14030.pdf
# https://github.com/microsoft/Swin-Transformer
# https://github.com/JingyunLiang/SwinIR
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.init as init

from .basic_ops import normalization

def set_module_trainable(module, trainable=True):
    if isinstance(module, nn.Module):
        for param in module.parameters():
            param.requires_grad = trainable
    elif isinstance(module, nn.Parameter):
        module.requires_grad = trainable

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LoRAFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, num_classes=0, rank=4, lora_alpha=1):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.num_classes = num_classes
        self.rank = rank
        self.lora_alpha = lora_alpha

        # Base FFN layer
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)  # (in, hidden)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)  # (hidden, out)
        self.act = act_layer()

        if num_classes > 0:
            # LoRA parameters
            self.fc1_lora_A = nn.Embedding(num_classes, self.in_features * self.rank)
            self.fc1_lora_B = nn.Embedding(num_classes, self.rank * self.hidden_features)
            self.fc2_lora_A = nn.Embedding(num_classes, self.hidden_features * self.rank)
            self.fc2_lora_B = nn.Embedding(num_classes, self.rank * self.out_features)
            self.scaling = self.lora_alpha / self.rank

            # Remove static freezing - now handled dynamically in forward()
            self._reset_parameters()

    def forward(self, x, label=None, lora=False):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        scaling_factor = self.scaling if lora else 0

        # assert label is not None
        assert (label >= 0).all() and (label < self.num_classes).all(), "Invalid label index"

        # set_module_trainable(self.fc1, not lora)
        # set_module_trainable(self.fc2, not lora)

        fc1_A = self.fc1_lora_A(label).view(B, self.in_features, self.rank)
        fc1_B = self.fc1_lora_B(label).view(B, self.rank, self.hidden_features)
        fc1_lora = torch.bmm(fc1_A, fc1_B) * scaling_factor
        # fc1_weight = self.fc1.weight.T + fc1_lora
        x = self.fc1(x) + torch.bmm(x, fc1_lora)
        
        x = self.act(x)

        fc2_A = self.fc2_lora_A(label).view(B, self.hidden_features, self.rank)
        fc2_B = self.fc2_lora_B(label).view(B, self.rank, self.out_features)
        fc2_lora = torch.bmm(fc2_A, fc2_B) * scaling_factor
        # fc2_weight = self.fc2.weight.T + fc2_lora
        x = self.fc2(x) + torch.bmm(x, fc2_lora)

        # Forward propagation
        # x = torch.matmul(x, fc1_weight) + self.fc1.bias
        # x = self.act(x)
        # x = torch.matmul(x, fc2_weight) + self.fc2.bias

        x = x.permute(0, 2, 1).view(B, self.out_features, H, W).contiguous()
        return x
    
    def _reset_parameters(self):
        """Initialize LoRA matrices with Kaiming initialization for A matrices and zero for B matrices."""
        if self.num_classes > 0:
            # Initialize fc1 LoRA A matrix (Kaiming uniform)
            # Shape: (num_classes, in_features * rank) -> (num_classes, in_features, rank)
            fc1_A = self.fc1_lora_A.weight.view(-1, self.in_features, self.rank)
            init.kaiming_uniform_(fc1_A, a=math.sqrt(5))  # PyTorch默认的nn.Linear初始化方式
            self.fc1_lora_A.weight.data = fc1_A.view(self.fc1_lora_A.weight.shape)

            # Initialize fc1 LoRA B matrix (zeros)
            # 原始权重保持冻结，LoRA初始扰动为0
            init.zeros_(self.fc1_lora_B.weight)

            # Initialize fc2 LoRA A matrix (Kaiming uniform)
            # Shape: (num_classes, hidden_features * rank) -> (num_classes, hidden_features, rank)
            fc2_A = self.fc2_lora_A.weight.view(-1, self.hidden_features, self.rank)
            init.kaiming_uniform_(fc2_A, a=math.sqrt(5))
            self.fc2_lora_A.weight.data = fc2_A.view(self.fc2_lora_A.weight.shape)

            # Initialize fc2 LoRA B matrix (zeros)
            init.zeros_(self.fc2_lora_B.weight)
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class WindowAttentionCondition(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of classes for LoRA conditioning.
        low_rank_dim (int): Rank of LoRA decomposition.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        single_bias (bool, optional): Whether to use a single bias across all channels.
    """

    def __init__(self, dim, window_size, num_heads, num_classes=10, low_rank_dim=4,
                 qkv_bias=True, qk_scale=None, single_bias=False, lora_alpha=1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.single_bias = single_bias
        self.low_rank_dim = low_rank_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.scaling = lora_alpha / self.low_rank_dim if low_rank_dim > 0 else 1.0

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))  # Wh*Ww, Wh*Ww

        # Base parameters
        self.qkv = nn.Parameter(torch.randn(dim, dim * 3))
        self.proj_out = nn.Parameter(torch.randn(dim, dim))

        # LoRA parameters
        if low_rank_dim > 0:
            self.qkv_A_emb = nn.Embedding(num_classes, dim * low_rank_dim)
            self.qkv_B_emb = nn.Embedding(num_classes, low_rank_dim * dim * 3)
            self.proj_A_emb = nn.Embedding(num_classes, dim * low_rank_dim)
            self.proj_B_emb = nn.Embedding(num_classes, low_rank_dim * dim)

            # Initialize LoRA and bias parameters
        self._reset_parameters()

        # Bias parameters
        if qkv_bias:
            self.bias = nn.Parameter(torch.zeros(dim * 3))
            self.bias_proj = nn.Parameter(torch.zeros(dim))
            if single_bias:
                self.bias_AB = nn.Embedding(num_classes, 3)
                self.bias_proj_AB = nn.Embedding(num_classes, 1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def _reset_parameters(self):
        """Initialize LoRA and bias parameters."""
        init.kaiming_uniform_(self.qkv, a=math.sqrt(5))
        init.kaiming_uniform_(self.proj_out, a=math.sqrt(5))

        # LoRA initialization
        if self.low_rank_dim > 0:
            for emb in [self.qkv_A_emb, self.proj_A_emb]:
                weight = emb.weight.view(emb.num_embeddings, self.dim, self.low_rank_dim)
                init.kaiming_uniform_(weight, a=math.sqrt(5))
                emb.weight.data = weight.view(emb.num_embeddings, -1)

            for emb in [self.qkv_B_emb, self.proj_B_emb]:
                emb.weight.data.zero_()

            # Bias initialization
            if self.single_bias:
                init.zeros_(self.bias_AB.weight)
                init.zeros_(self.bias_proj_AB.weight)

    def forward(self, x, label, mask=None, lora=False):
        """
        Args:
            x: Input features with shape (B, C, H, W)
            label: Class labels with shape (B,)
            mask: Optional mask with shape (num_windows, Wh*Ww, Wh*Ww)
        """
        B, C, Ph, Pw = x.shape
        N = self.window_size[0] * self.window_size[1]
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N, C)

        # set_module_trainable(self.qkv, not lora)
        # set_module_trainable(self.proj_out, not lora)

        if self.low_rank_dim > 0:
            qkv = self._apply_lora(x, label, self.qkv, self.qkv_A_emb, self.qkv_B_emb, mode=lora)
        else:
            qkv = self._apply_lora(x, base_weight=self.qkv)

        if self.qkv_bias:
            bias = self.bias.view(1, 1, -1)
            qkv = qkv + bias
            if self.single_bias:
                bias_AB = self.bias_AB(label).view(B, 1, -1)
                qkv = qkv + bias_AB  # (B, N, 3C)

        qkv = qkv.permute(0, 2, 1).reshape(B, -1, Ph, Pw)  # (B, 3C, H, W)
        qkv = window_partition(qkv, self.window_size[0])  # (num_windows * B, Wh, Ww, 3C)
        B_ = qkv.shape[0]
        qkv = qkv.view(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).contiguous()  # (B_, H, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], -1, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).contiguous().view(B_, N, C)  # (B_, N_, C)

        # Merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], Ph, Pw)  # (B, H, W, C)
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # (B, N, C)

        # Apply LoRA to proj_out
        if self.low_rank_dim > 0:
            x = self._apply_lora(x, label, self.proj_out, self.proj_A_emb, self.proj_B_emb, mode=lora)
        else:
            x = self._apply_lora(x, base_weight=self.proj_out)

        if self.qkv_bias:
            bias = self.bias_proj.view(1, 1, -1)
            x = x + bias
            if self.single_bias:
                bias_AB = self.bias_proj_AB(label).view(B, 1, -1)
                x = x + bias_AB

        x = x.permute(0, 2, 1).view(B, C, Ph, Pw).contiguous()
        return x

    def _apply_lora(self, x, label=None, base_weight=None, A_emb=None, B_emb=None, mode=None):
        """
        Apply LoRA adjustment to the base weight matrix.
        Args:
            x: Input tensor (B, N, C)
            label: Class label (B,)
            base_weight: Base weight matrix (C, 3C)
            A_emb: LoRA A embedding (num_classes, C * low_rank_dim)
            B_emb: LoRA B embedding (num_classes, low_rank_dim * 3C)
        Returns:
            Adjusted output (B, N, 3C)
        """
        B, N, C = x.shape
        if label is not None:
            C_base = base_weight.size(0)  # C

            scaling_factor = self.scaling if mode else 0

            # Get LoRA parameters
            A = A_emb(label).view(B, C_base, self.low_rank_dim)  # (B, C, r)
            B = B_emb(label).view(B, self.low_rank_dim, -1)  # (B, r, 3C)
            lora_weight = torch.bmm(A, B) * scaling_factor  # (B, C, 3C)

            # Combine base weight and LoRA
            weight = base_weight + lora_weight # (B, C, 3C)
        else:
            weight = base_weight.unsqueeze(0).expand(B, -1, -1)

        # Apply linear transformation
        return torch.bmm(x, weight)  # (B, N, 3C)
    

   
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), B_ x H x N x C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerConditionBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=normalization, num_classes=10, low_rank_dim=4, single_bias=False, lora_alpha=1):
        super().__init__()
        self.dim = dim   # 192
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionCondition(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, num_classes=num_classes, low_rank_dim=low_rank_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, single_bias=single_bias, lora_alpha=lora_alpha)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = LoRAFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, num_classes=num_classes, rank=low_rank_dim, lora_alpha=lora_alpha)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, 1, H, W))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        # mask_windows = window_partition(img_mask, self.window_size).permute(0,2,3,1).contiguous()  # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, window_size*window_size

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, label, lora):
        '''
        Args:
            x: B x C x Ph x Pw, Ph = H // patch_size   下采样四倍后的
        Out:
            x: B x (H*W) x C
        '''
        B, C, Ph, Pw = x.shape
        # x_size = (Ph, Pw)
        x_size = [Ph, Pw] # input_resolution是列表

        shortcut = x
        x = self.norm1(x)    # B x C x Ph x Pw

        # cyclic shift, shifted_x: B x C x Ph x Pw
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        if self.input_resolution == x_size:
            # attn_windows = self.attn(x_windows, mask=self.attn_mask.to(x.dtype))  # (NW*B) x (Ws*Ws) x C
            shifted_x = self.attn(shifted_x, label=label, mask=self.attn_mask, lora=lora)  # (NW*B) x (Ws*Ws) x C
        else:
            shifted_x = self.attn(shifted_x, label=label, mask=self.calculate_mask(x_size).to(x.device, x.dtype), lora=lora)

        # partition windows, NW: number of windows, Ws: window size
        # x_windows = window_partition(shifted_x, self.window_size)  # (NW*B) x Ws x Ws x C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (NW*B) x (Ws*Ws) x C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # if self.input_resolution == x_size:
        #     # attn_windows = self.attn(x_windows, mask=self.attn_mask.to(x.dtype))  # (NW*B) x (Ws*Ws) x C
        #     attn_windows = self.attn(shifted_x, label=label, window_size=self.window_size, mask=self.attn_mask)  # (NW*B) x (Ws*Ws) x C
        # else:
        #     attn_windows = self.attn(shifted_x, label=label, window_size=self.window_size, mask=self.calculate_mask(x_size).to(x.device, x.dtype))

        # # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, Ph, Pw)  # B x C x Ph x Pw

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x), label, lora))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=normalization):
        super().__init__()
        self.dim = dim   # 192
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, 1, H, W))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        # mask_windows = window_partition(img_mask, self.window_size).permute(0,2,3,1).contiguous()  # nW, window_size, window_size, 1
        mask_windows = window_partition(img_mask, self.window_size).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # nW, window_size*window_size

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        '''
        Args:
            x: B x C x Ph x Pw, Ph = H // patch_size   下采样四倍后的
        Out:
            x: B x (H*W) x C
        '''
        B, C, Ph, Pw = x.shape
        # x_size = (Ph, Pw)
        x_size = [Ph, Pw] # input_resolution是列表

        shortcut = x
        x = self.norm1(x)    # B x C x Ph x Pw

        # cyclic shift, shifted_x: B x C x Ph x Pw
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows, NW: number of windows, Ws: window size
        x_windows = window_partition(shifted_x, self.window_size)  # (NW*B) x Ws x Ws x C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (NW*B) x (Ws*Ws) x C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            # attn_windows = self.attn(x_windows, mask=self.attn_mask.to(x.dtype))  # (NW*B) x (Ws*Ws) x C
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (NW*B) x (Ws*Ws) x C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device, x.dtype))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Ph, Pw)  # B x C x Ph x Pw

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size (int): image resolution. Defaulr: 224
        patch_size (int): patch resolution. Default: 1
        patch_norm (bool): patch normalization. Default: False
    """
    def __init__(
            self,
            in_chans,
            embed_dim,
            num_heads,
            window_size,
            depth=2,
            img_size=224,
            patch_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=normalization,
            use_checkpoint=False,
            patch_norm=True,
             ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                patch_norm=patch_norm,
                )
        num_patches = self.patch_embed.num_patches
        input_resolution = self.patch_embed.patches_resolution
        self.input_resolution = input_resolution

        self.patch_unembed = PatchUnEmbed(
                out_chans=in_chans,
                embed_dim=embed_dim,
                patch_norm=patch_norm,
                )

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                        dim=embed_dim,  # 192
                        input_resolution=input_resolution,  # 除以4之后的
                        num_heads=num_heads,  # 6
                        window_size=window_size,  # 8
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,  # 4
                        qkv_bias=qkv_bias,  # True
                        qk_scale=qk_scale,  # None
                        drop=drop,  # 0
                        attn_drop=attn_drop,  # 0
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # 0
                        norm_layer=norm_layer,  # GN32
                                 )
            for i in range(depth)])

    def forward(self, x):
        '''
        Args:
            x: B x C x H x W, H,W: height and width after patch embedding
            x_size: (H, W)
        Out:
            x: B x H x W x C
        '''
        x = self.patch_embed(x) # B x embed_dim x Ph x Pw   下采样四倍
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_unembed(x) # B x C x Ph x Pw
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicConditionLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size (int): image resolution. Defaulr: 224
        patch_size (int): patch resolution. Default: 1
        patch_norm (bool): patch normalization. Default: False
    """
    def __init__(
            self,
            in_chans,
            embed_dim,
            num_heads,
            window_size,
            depth=2,
            img_size=224,
            patch_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            drop_path=0.,
            norm_layer=normalization,
            use_checkpoint=False,
            patch_norm=True,
            num_clsses=10,
            low_rank_dim=4,
            single_bias=False,
            lora_alpha=1,
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                patch_norm=patch_norm,
                )
        num_patches = self.patch_embed.num_patches
        input_resolution = self.patch_embed.patches_resolution
        self.input_resolution = input_resolution

        self.patch_unembed = PatchUnEmbed(
                out_chans=in_chans,
                embed_dim=embed_dim,
                patch_norm=patch_norm,
                )

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerConditionBlock(
                        dim=embed_dim,  # 192
                        input_resolution=input_resolution,  # 除以4之后的
                        num_heads=num_heads,  # 6
                        window_size=window_size,  # 8
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,  # 4
                        qkv_bias=qkv_bias,  # True
                        qk_scale=qk_scale,  # None
                        drop=drop,  # 0
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # 0
                        norm_layer=norm_layer,  # GN32
                        num_classes=num_clsses,
                        low_rank_dim=low_rank_dim,
                        single_bias=single_bias,
                        lora_alpha=lora_alpha,
                                 )
            for i in range(depth)])

    def forward(self, x, label, lora):
        '''
        Args:
            x: B x C x H x W, H,W: height and width after patch embedding
            x_size: (H, W)
        Out:
            x: B x H x W x C
        '''
        x = self.patch_embed(x) # B x embed_dim x Ph x Pw   下采样四倍
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, label)
            else:
                x = blk(x, label, lora)
        x = self.patch_unembed(x) # B x C x Ph x Pw
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        patch_norm (bool, optional): True, GroupNorm32
        in_chans (int): unused. Number of input image channels. Default: 3.
    """
    def __init__(
            self,
            in_chans,
            img_size=224,
            patch_size=4,   # patch大小为4*4，一个patch内有16个像素
            embed_dim=96,
            patch_norm=False,
            ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)   # 4 -> (4,4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]   # 算一共有多少张patch
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)    # 下采样4倍, 扩充维度
        if patch_norm:   # false
            self.norm = normalization(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: B x C x H x W
        output: B x embed_dim x Ph x Pw, Ph = H // patch_size

        """
        x = self.proj(x)  # B x embed_dim x Ph x Pw
        x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Patch to Image.

    Args:
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, out_chans, embed_dim=96, patch_norm=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)
        if patch_norm:
            self.norm = normalization(out_chans)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        '''
        Args:
            x: B x C x Ph x Pw
        out: B x C x Ph x Pw
        '''
        x = self.norm(self.proj(x))
        return x

    def flops(self):
        flops = 0
        return flops

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
