# -*- coding: utf-8 -*-
"""
transformer.py — MSCAS-AIFI 多尺度卷积加法自注意力模块（毕业论文第三章改进点2）

本文件为精简版，仅保留 FMA1.yaml 实际使用的 AIFI_CAS_att3 模块及其依赖组件：
  - SpatialOperation_att3:    多尺度空间注意力（MSSA），3×3 + 空洞5×5双分支
  - ChannelOperation_att3:    高效通道注意力（ECA），1×1 Conv 降维升维
  - MultiScaleDWConv:         多尺度深度卷积，3×3 + 空洞5×5 加权融合
  - AdditiveTokenMixer_att3:  加法 Token 混合器，O(N) 复杂度替代 MHSA
  - AIFI_CAS_att3:            MSCAS-AIFI 完整模块（Pre-Norm + DropPath + DWFFN）
"""

import torch
import torch.nn as nn

__all__ = ['SpatialOperation_att3', 'ChannelOperation_att3', 'MultiScaleDWConv',
           'AdditiveTokenMixer_att3', 'AIFI_CAS_att3']


# ============================================================================
# SpatialOperation_att3 — 多尺度空间注意力（MSSA）
# 3×3 DWConv + 空洞5×5 DWConv（dilation=2）双分支，拼接后生成空间注意力图，
# 激活函数使用 SiLU（梯度更平滑，有利于小目标特征保留）。
# ============================================================================

class SpatialOperation_att3(nn.Module):
    """多尺度空间注意力（MSSA）"""
    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU(True),
        )
        # Branch 2: Dilated Convolution (Effective 5x5)
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 2, groups=dim, dilation=2),
            nn.BatchNorm2d(dim),
            nn.SiLU(True),
        )
        self.fuse = nn.Conv2d(dim * 2, 1, 1, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        feat = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        attention = self.fuse(feat)
        return x * self.act(attention)


# ============================================================================
# ChannelOperation_att3 — 高效通道注意力（ECA）
# 自适应平均池化 → 1×1 Conv 降维 → SiLU → 1×1 Conv 升维 → Sigmoid 门控。
# ============================================================================

class ChannelOperation_att3(nn.Module):
    """高效通道注意力（ECA）"""
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.SiLU(True),
            nn.Conv2d(dim // 4, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# ============================================================================
# MultiScaleDWConv — 多尺度深度卷积（MSDW）
# 3×3 + 空洞5×5（dilation=2）双分支加权融合，捕获不同尺度的背景纹理。
# ============================================================================

class MultiScaleDWConv(nn.Module):
    """多尺度深度卷积（MSDW）"""
    def __init__(self, dim):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, 3, 1, 2, groups=dim, dilation=2)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(True)
        self.fuse_weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        out = self.bn(self.fuse_weight * x1 + (2 - self.fuse_weight) * x2)
        return self.act(out)


# ============================================================================
# AdditiveTokenMixer_att3 — 加法 Token 混合器
# 替代传统 Transformer 的矩阵乘法注意力，采用加法混合策略实现 O(N) 复杂度。
# 流程: Q/K 经 MSSA+ECA 双重增强 → 加法融合(Q+K) → MSDW 补充局部 → 加权V → 分组投影
# ============================================================================

class AdditiveTokenMixer_att3(nn.Module):
    """加法 Token 混合器（MSCAS-AIFI 的核心组件）

    替代传统 Transformer 的矩阵乘法注意力，采用加法混合策略实现 O(N) 复杂度的全局 Token 交互。
    流程：
    1. 1×1卷积将输入映射为 Q、K、V 三分支（hidden_ratio 压缩通道降低计算量）；
    2. Q 经多尺度空间注意力(MSSA)增强局部边缘纹理特征，K 经高效通道注意力(ECA)突出高响应通道；
    3. 加法融合(Q+K)替代矩阵乘法(Q×K^T)，再经多尺度深度卷积(MSDW)补充局部特征；
    4. 融合结果与 V 逐元素加权，经分组卷积投影回原维度。
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0., hidden_ratio=0.5):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)

        # 计算合适的分组数
        if hidden_dim <= 32:
            groups = hidden_dim
        else:
            groups = hidden_dim // 8
            while hidden_dim % groups != 0 and groups > 1:
                groups -= 1

        self.qkv = nn.Conv2d(dim, 3 * hidden_dim, 1, stride=1, padding=0, bias=attn_bias)

        # Q/K 分别经空间+通道注意力增强
        self.oper_q = nn.Sequential(
            SpatialOperation_att3(hidden_dim),
            ChannelOperation_att3(hidden_dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation_att3(hidden_dim),
            ChannelOperation_att3(hidden_dim),
        )

        self.dwc = MultiScaleDWConv(hidden_dim)

        # 投影层：hidden_dim → dim，分组卷积加速
        if dim % groups != 0:
            groups = 1
        self.proj = nn.Conv2d(hidden_dim, dim, 3, 1, 1, groups=groups)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = self.oper_q(q)
        k = self.oper_k(k)

        # 核心交互：加法融合后的特征调制 V
        attn_map = self.dwc(q + k)
        out = self.proj(attn_map * v)

        out = self.proj_drop(out)
        return out


# ============================================================================
# AIFI_CAS_att3 — MSCAS-AIFI 完整模块（改进点2）
# 替代 RT-DETR 原始 AIFI 的 MHSA，采用 Pre-Norm + DropPath + DWFFN 结构。
# 将复杂度从 O(N²) 降至 O(N)，在极低计算量下实现全局 Token 高效交互。
# ============================================================================

class AIFI_CAS_att3(nn.Module):
    """MSCAS-AIFI 模块（改进点2：混合编码器注意力替换）

    替代 RT-DETR 原始 AIFI 模块中的传统多头自注意力(MHSA)，采用多尺度卷积加法自注意力机制。
    通过加法 Token 混合替代矩阵乘法注意力计算，将复杂度从 O(N²) 降至 O(N)，
    在极低计算量下实现全局 Token 高效交互，精准捕捉小目标与背景的全局关联关系，
    同时抑制背景噪声、突出目标核心特征。

    优化点：
    1. 使用 Pre-Norm 结构（先 norm 再计算），利于深层网络训练和小目标梯度回传；
    2. 引入 DropPath (Stochastic Depth)，增强泛化能力；
    3. FFN 结合深度卷积(DWConv)，增强局部特征提取能力；
    4. 移除 num_heads 硬编码，适配灵活配置。
    """
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0,
                 act=nn.GELU(), normalize_before=True, mlp_ratio=4., drop_path=0.):
        super().__init__()

        # 1. Token Mixing 部分
        self.additivetoken = AdditiveTokenMixer_att3(c1, hidden_ratio=0.5)

        # 2. FFN 部分（带深度卷积）
        hidden_dim = int(c1 * mlp_ratio)
        self.fc1 = nn.Conv2d(c1, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, c1, 1)

        # 3. Normalization
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)

        # 4. Dropout & DropPath
        self.dropout = nn.Dropout(dropout)
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.DropPath(drop_path)

        self.act = act
        self.normalize_before = normalize_before

    def forward_pre(self, src):
        """Pre-Norm 结构: x = x + DropPath(Attn(Norm(x)))"""
        B, C, H, W = src.shape
        src = src.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Attention Branch
        src2 = self.norm1(src)
        src2 = src2.transpose(1, 2).view(B, C, H, W)
        src2 = self.additivetoken(src2)
        src2 = src2.flatten(2).transpose(1, 2)
        src = src + self.drop_path(src2)

        # FFN Branch
        src2 = self.norm2(src)
        src2 = self.fc1(src2.transpose(1, 2).view(B, -1, H, W)).flatten(2).transpose(1, 2)
        src2 = self.act(src2)

        # DWConv 需要在 Conv 格式下
        src2 = src2.transpose(1, 2).view(B, -1, H, W)
        src2 = self.dwconv(src2)
        src2 = self.dropout(src2)

        src2 = self.fc2(src2).flatten(2).transpose(1, 2)
        src = src + self.drop_path(src2)

        return src.transpose(1, 2).view(B, C, H, W)

    def forward_post(self, src):
        """Post-Norm 结构（保留作为兼容，默认不使用）"""
        B, C, H, W = src.shape
        src = src.flatten(2).transpose(1, 2)

        src2 = self.additivetoken(src.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        src = self.norm1(src + self.drop_path(src2))

        src2 = self.fc2(
            self.dropout(self.act(self.fc1(src.transpose(1, 2).view(B, -1, H, W)))).flatten(2).transpose(1, 2))
        src = self.norm2(src + self.drop_path(src2))

        return src.transpose(1, 2).view(B, C, H, W)

    def forward(self, src):
        if self.normalize_before:
            return self.forward_pre(src)
        return self.forward_post(src)
