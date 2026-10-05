# -*- coding: utf-8 -*-
"""
block.py — 红外小型无人机目标检测自定义模块

本文件为精简版，仅保留 FMA1.yaml 实际使用的核心改进模块：
  - ADown:              分路异构下采样（改进点1a）
  - C2f_FAT:            C2f + FAT_Block 多尺度窗口注意力（改进点1b）
  - Partial_conv3:      部分卷积（Faster_Block 的核心组件）
  - Faster_Block:      部分卷积快速块（FMA 的核心组件）
  - MANet:              多分支聚合网络（改进点3核心结构）
  - MANet_FasterBlock:  FMA 快速聚合网络（改进点3最终使用）
"""

import torch
import torch.nn as nn
from timm.layers import DropPath

from ..modules.conv import Conv, DWConv
from ..modules.block import C2f, Bottleneck
from .filc import FAT_Block

__all__ = ['Partial_conv3', 'Faster_Block', 'ADown', 'MANet', 'MANet_FasterBlock', 'C2f_FAT']


# ============================================================================
# Partial_conv3 — 部分卷积（FasterNet 核心组件）
# 采用"通道分治"策略：仅对部分通道执行 3×3 卷积，其余通道保持不变，
# 既降低计算开销，又避免背景噪声通过全通道扩散。
# ============================================================================

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


# ============================================================================
# Faster_Block — 部分卷积快速块（FastNet）
# 结合 Partial_conv3 空间混合与 MLP 非线性增强，通过残差连接保留小目标特征。
# ============================================================================

class Faster_Block(nn.Module):
    def __init__(self, inc, dim, n_div=4, mlp_ratio=2, drop_path=0.1,
                 layer_scale_init_value=0.0, pconv_fw_type='split_cat'):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


# ============================================================================
# ADown — 分路异构下采样（改进点1a）
# 先平均池化弱化背景杂波，再通道拆分双分支：
#   分支1: 3×3卷积(stride=2)保留小目标边缘纹理
#   分支2: 3×3最大池化(stride=2)强化亮度对比度 + 1×1卷积调整通道
# ============================================================================

class ADown(nn.Module):
    """ADown 分路异构下采样模块（改进点1a：骨干网络低损下采样）

    针对红外小目标像素占比极低、常规下采样易丢失微弱特征的问题，采用分路异构下采样策略：
    1. 先对输入特征图执行步幅1的平均池化，弱化背景杂波噪声；
    2. 将特征通道拆分为两部分：
       - 分支1: 3×3卷积(步幅2)完成下采样，保留小目标边缘纹理特征；
       - 分支2: 3×3最大池化(步幅2)强化小目标亮度对比度，再1×1卷积调整通道；
    3. 两分支沿通道拼接，在完成分辨率下采样与通道数扩增的同时，最大限度减少小目标特征损失。
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)  # 分支1: 3×3卷积下采样，保留边缘纹理
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)  # 分支2: 1×1卷积调整通道

    def forward(self, x):
        # 步幅1平均池化，弱化背景杂波噪声
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        # 通道拆分为两路并行处理
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)  # 分支1: 3×3卷积下采样
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  # 分支2: 最大池化强化亮度对比
        x2 = self.cv2(x2)  # 分支2: 1×1卷积调整通道
        return torch.cat((x1, x2), 1)  # 通道拼接输出


# ============================================================================
# MANet — 多分支聚合网络（改进点3核心结构）
# 四分支特征解耦：
#   分支1 (y0): 1×1卷积提取通道级核心特征
#   分支2 (y1): 1×1→非对称DWConv→1×1 强化空间纹理特征
#   分支3/4 (y2/y3): 直接通道拆分，保留原始特征基础信息
#   额外Bottleneck分支: 串行特征提取
# ============================================================================

class MANet(nn.Module):
    """MANet 多分支聚合网络（改进点3的核心结构）

    替换 RT-DETR 原始 RepC3 模块，采用多分支特征解耦设计：
    - 分支1 (y0): 1×1卷积提取通道级核心特征，聚焦小目标高响应通道；
    - 分支2 (y1): 1×1→非对称DWConv→1×1，强化空间纹理特征（边缘、亮度梯度）；
    - 分支3/4 (y2/y3): 直接通道拆分，保留原始特征基础信息，避免卷积过度压制微弱特征；
    - 额外Bottleneck分支: 串行特征提取，增强特征表达能力。
    最终所有分支沿通道拼接，经1×1卷积聚合输出。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                       Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))


class MANet_FasterBlock(MANet):
    """MANet + FasterBlock 变体（FMA 模块实际使用的结构）

    在 MANet 多分支架构基础上，将 Bottleneck 替换为 Faster_Block（部分卷积 FastNet），
    采用"通道分治"策略：仅对部分通道执行3×3卷积提取空间特征，其余通道保持不变，
    既降低计算开销，又避免背景噪声通过全通道扩散，实现轻量化与精度的协同优化。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))


# ============================================================================
# C2f-FAT — C2f + FAT_Block（改进点1b）
# 在 C2f 残差结构基础上，将核心特征建模分支替换为 FAT_Block，
# 通过 FDWA 多尺度窗口注意力与 FMFFN 频率调制增强小目标特征显著性。
# ============================================================================

class C2f_FAT(C2f):
    """C2f-FAT 模块（改进点1b：骨干网络特征增强）

    在 C2f 残差结构基础上，将核心特征建模分支替换为 FAT_Block。
    FAT_Block 包含两个核心组件：
    1. FDWA (Feature Dilated Window Attention): 多尺度窗口注意力，聚焦小目标区域；
    2. FMFFN (Frequency Modulated Feed-Forward Network): 窗口内频率调制，压制背景低频噪声。
    通过残差连接将增强特征与原始特征融合，确保目标特征被放大并保留。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FAT_Block(self.c) for _ in range(n))
