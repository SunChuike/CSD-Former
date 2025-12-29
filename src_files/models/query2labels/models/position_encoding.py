# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and simplify the model.
"""
import math
import torch
from torch import nn
from torch.functional import Tensor

# from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # maxH 和 maxW 定义了预计算编码的最大尺寸
        self.maxH = maxH
        self.maxW = maxW
        
        # 预计算一个足够大的位置编码 buffer
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        # 创建一个 (1, maxH, maxW) 的张量作为基础
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            # 归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 计算不同维度的除数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 应用 sin/cos 变换
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # 将 x 和 y 方向的编码拼接起来
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    # --- 【核心修改点】 ---
    def forward(self, input: Tensor):
        """
        根据输入张量的实际 H, W 尺寸，从预计算的 pe buffer 中裁剪出相应部分。
        """
        # 获取输入特征图的实际 H 和 W
        # input 的形状是 [Batch, Channels, H, W]
        h, w = input.shape[-2:]

        # 确保输入尺寸不超过预计算的最大尺寸
        if h > self.maxH or w > self.maxW:
            raise ValueError(
                f"Input feature map size ({h}x{w}) exceeds the maximum size "
                f"of pre-computed position embeddings ({self.maxH}x{self.maxW}). "
                "Consider increasing maxH and maxW during initialization."
            )

        # 从 self.pe 中裁剪出需要的 [1, C, H, W] 部分
        # 然后在 batch 维度上复制
        pos_embedding = self.pe[:, :, :h, :w].repeat((input.size(0), 1, 1, 1))
        
        return pos_embedding

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2

    # --- 【逻辑调整点】 ---
    # 对于 ViT 类型的 backbone，下采样率是 patch_size
    # 对于 DINOv3 ViT-B/16, patch_size 是 16
    if args.backbone.startswith('dinov3'):
        # 假设 patch size 可以从模型名称中推断，这里硬编码为 16
        # 更稳健的方法是从 args 或模型配置中获取
        patch_size = 16 
        downsample_ratio = patch_size
    elif args.backbone in ['CvT_w24']:
        downsample_ratio = 16
    else: # 默认是为 ResNet 等 CNN 设计的
        downsample_ratio = 32
    
    # 计算期望的特征图尺寸
    expected_h = args.image_size // downsample_ratio
    expected_w = args.image_size // downsample_ratio

    if args.position_embedding in ('v2', 'sine'):
        # 使用计算出的期望尺寸来初始化 PositionEmbeddingSine
        # maxH 和 maxW 作为 buffer 的最大尺寸
        position_embedding = PositionEmbeddingSine(
            N_steps, 
            normalize=True, 
            maxH=expected_h, 
            maxW=expected_w
        )
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding