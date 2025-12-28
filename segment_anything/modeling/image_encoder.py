# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock
import numpy as np

def generate_gaussian_heatmap(shape, points, sigma, output_dir='data/point_img/embedding_gas/heatmaps'):
    """
    生成多个点的高斯热图并保存到指定目录
    :param shape: 热图的尺寸 (height, width)
    :param points: 包含多个中心坐标的列表 [(x1, y1), (x2, y2), ...]
    :param sigma: 高斯分布的标准差
    :param output_dir: 输出保存热图的目录
    :return: 高斯热图 (torch.Tensor)
    """
    # 创建输出目录（如果不存在的话）
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成网格
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x = x.to('cuda')
    y = y.to('cuda')
    
    # 初始化热图
    # heatmap = torch.zeros(shape, dtype=torch.float32, device='cpu')
    heatmap = torch.zeros(shape, dtype=torch.float32, device='cuda')
    
    # 计算多个点的高斯分布并叠加
    for center in points:
        dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
        heatmap += torch.exp(-dist_sq / (2 * sigma ** 2))
    
    # 归一化热图
    heatmap = torch.clamp(heatmap, 0, 1)
    
    return heatmap

def boundary_consistency_loss(F_s, F_d, M, eps=1e-8):
    """
    计算 Boundary-Consistent Alignment Loss（基于皮尔逊相关系数）

    参数:
        F_s: 浅层特征 (B, H, W, C)
        F_d: 深层特征 (B, H, W, C)
        M: 锚点 attention map (B, 1, H, W)
    返回:
        loss: scalar（越小越好，负相关系数）
    """
    M = M.float()

    # 保证 shape 一致 [B, C, H, W]
    F_s = F_s.permute(0, 3, 1, 2)
    F_d = F_d.permute(0, 3, 1, 2)

    # 乘上 mask，保留边缘区域
    F_s_b = F_s * M  # (B, C, H, W)
    F_d_b = F_d * M

    # reshape 成 (B, C, H*W)
    B, C, H, W = F_s_b.shape
    F_s_flat = F_s_b.view(B, C, -1)  # (B, C, N)
    F_d_flat = F_d_b.view(B, C, -1)

    # 计算每个 batch、每个通道上的皮尔逊相关系数
    mean_s = F_s_flat.mean(dim=-1, keepdim=True)
    mean_d = F_d_flat.mean(dim=-1, keepdim=True)

    F_s_centered = F_s_flat - mean_s
    F_d_centered = F_d_flat - mean_d

    numerator = (F_s_centered * F_d_centered).sum(dim=-1)  # (B, C)
    denominator = (
        F_s_centered.pow(2).sum(dim=-1).sqrt() * F_d_centered.pow(2).sum(dim=-1).sqrt() + eps
    )  # (B, C)

    corr = numerator / denominator  # (B, C)

    # 取 1 - corr 作为 loss（越相关越小）
    loss_per_channel = 1 - corr  # (B, C)

    # 对 batch 和 channel 求平均
    loss = loss_per_channel.mean()

    return loss

class Anchor(nn.Module):
    def __init__(self, alpha=1):
        super(Anchor, self).__init__()
        self.alpha = alpha
    
    def forward(self, shallow_feature):
        shallow_feature_permuted = shallow_feature.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Sobel 
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=shallow_feature.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=shallow_feature.device).view(1, 1, 3, 3)
        grad_x = F.conv2d(shallow_feature_permuted[:, :1, :, :], sobel_x, padding=1)
        grad_y = F.conv2d(shallow_feature_permuted[:, :1, :, :], sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x**2 + grad_y**2)
        edge_map = edge_map / (edge_map.max() + 1e-8)  # [0, 1]

        anchor_map = self.alpha * edge_map
        print("anchor_map.shape:", anchor_map.shape)

        return  anchor_map

class AnchorQualityChecker:
    def __init__(self, threshold_ratio=1.5, min_value=0.1):
        """
        :param threshold_ratio: box 内与外部响应比值超过这个阈值才认为是高质量
        :param min_value: box 内平均值至少大于该值
        """
        self.threshold_ratio = threshold_ratio
        self.min_value = min_value

    def is_high_quality(self, anchor_map, box):
        """
        判断 anchor_map 是否为高质量边缘图

        :param anchor_map: Tensor, shape=(1, 1, H, W) 或 (1, H, W)
        :param box: (x1, y1, x2, y2) 形式的整数坐标
        :return: bool 是否为高质量
        """
        if anchor_map.ndim == 4:
            anchor_map = anchor_map.squeeze(0).squeeze(0)  # -> (H, W)
        elif anchor_map.ndim == 3:
            anchor_map = anchor_map.squeeze(0)  # -> (H, W)

        x1, y1, x2, y2 = box
        H, W = anchor_map.shape

        # 防止越界
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)

        # 提取 box 区域和非 box 区域
        box_region = anchor_map[y1:y2+1, x1:x2+1]
        mask = torch.ones_like(anchor_map, dtype=torch.bool)
        mask[y1:y2+1, x1:x2+1] = False
        outside_region = anchor_map[mask]

        # 计算均值（也可以用 max）
        box_mean = box_region.mean().item()
        outside_mean = outside_region.mean().item()

        print(f"[AnchorChecker] box_mean={box_mean:.4f}, outside_mean={outside_mean:.4f}")

        if box_mean > self.min_value and box_mean > self.threshold_ratio * outside_mean:
            return True
        else:
            return False

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,#图像的通道数，RGB图像
        embed_dim: int = 768,#每个patch的嵌入维度
        depth: int = 12,#VIT的深度，即堆叠的Block的块的数量
        num_heads: int = 12,#每个VIT中头的数量，只关注一个
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,#在三个特征矩阵上添加可学习的偏置
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size=patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        #nn.Parameter定义模型参数的类
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            #每个patch的第一张图像；patch数量；patch数量；每个patch的嵌入维度（Patch embedding dimension）
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()#创建self.blocks空间，其被用来存储多个 Block 实例
        # 计算四等分的位置索引
        quarter_indices = {(depth // 4)-1, (depth // 2)-1, (3 * depth // 4)-1, depth - 1}

        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )

            self.blocks.append(block)
            
        self.anchor = Anchor()
        self.anchorChecker = AnchorQualityChecker()
        self.shallow_feature = None
        self.deep_feature = None
        self.anchor_map = None

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
        self.heatmap_proj = nn.Linear(1, self.embed_dim)  # 将1维投影为768维

    def forward(self,
                x: torch.Tensor, 
                input_points: torch.Tensor = None, 
                input_box: torch.Tensor = None, 
                feature_map: torch.Tensor = None,
    ) -> torch.Tensor:
        # print("x.device in image_encoder first:", x.device)
        # print("Positional embedding shape:", self.pos_embed.shape)
        # print("patch num:", self.img_size//self.patch_size)
        # print("x before patch_embed:", x.shape)
        # print("x.max before conv:", torch.max(x))
        # print("x.min before conv:", torch.min(x))
        x = self.patch_embed(x)
        # 嵌入点信息
        # print("x.max after conv:", torch.max(x))
        # print("x.min after conv:", torch.min(x))
        # position id
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # 将prompt输入进input
        if input_points!= None and input_box==None and feature_map==None:
            # point
            point_tensor =  torch.zeros(1, self.img_size // self.patch_size, self.img_size // self.patch_size, self.embed_dim, dtype=torch.float32)
            # # 将创建的张量移到cuda上
            point_tensor = point_tensor.to('cuda')
            print("input_points:", input_points)

            #使用高斯热图
            sigma = 2  # 高斯标准差，决定扩散范围

            # 位置增强
            heatmap = generate_gaussian_heatmap((64, 64), input_points, sigma)
            # # 扩大高值区域
            # heatmap = heatmap ** 0.01
            
            # 假设 heatmap 是 (64, 64)
            heatmap = heatmap.unsqueeze(-1).expand(-1, -1, 768)  # -> (64, 64, 768)
            heatmap = heatmap.unsqueeze(0)
            
            point_tensor += heatmap
            
            # print("point_tensor.shape after heatmap:", point_tensor.shape)
            max_value = torch.max(heatmap)
            min_value = torch.min(heatmap)
            # print("man_value in heatmap:", max_value)
            # print("min_value in heatmap:", min_value)

            # 限制热图值的最大值，防止数值溢出
            point_tensor = torch.clamp(point_tensor, max=1.0)

            # 计算 block 处插入的索引 注意是从0开始
            first = 0
            quarter = len(self.blocks) // 4
            half = len(self.blocks) // 2 
            three_quarters = 3 * len(self.blocks) // 4
            all = len(self.blocks)
                
            # 位置增强
            for i, blk in enumerate(self.blocks):
                if i == 5:  # 第6层（索引从0开始）
                    self.shallow_feature = x  # 形状: (B, H, W, C) = (1, 64, 64, 768)
                    self.anchor_map = self.anchor(self.shallow_feature)
                    if self.anchor_map == None:
                        print("Edge Anchor Wrong!")
                    
                if i in [first, quarter, half, three_quarters]:
                    print("i:", i)
                    x = x + point_tensor  # 在1/4, 2/4, 3/4处添加point_tensor
                    
                x = blk(x)
                
            self.deep_feature = x
                    
        elif input_points==None and input_box!=None and feature_map==None:
            print("box!")
            # box
            box_tensor =  torch.zeros(1, self.img_size // self.patch_size, self.img_size // self.patch_size, self.embed_dim, dtype=torch.float32)
            heatmap = torch.zeros(1, self.img_size // self.patch_size, self.img_size // self.patch_size)
            box_tensor = box_tensor.to('cuda')
            heatmap = heatmap.to('cuda')
            x1, y1, x2, y2 = input_box
            print("x1y1x2y2:", x1, y1, x2, y2)
            
            # 1. 计算中心点和最短边
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            min_half_size = min(x2 - x1, y2 - y1) / 2
            max_half_size = max(x2 - x1, y2 - y1) / 2
            #  设置 sigma 为最长斜边一半的 0.5 倍，可调节平滑度
            diagonal = (min_half_size ** 2 + max_half_size ** 2) ** 0.5
            sigma = diagonal * 0.5
            # 3. 生成高斯热图（输入 shape 是 H x W）
            heatmap = generate_gaussian_heatmap((64, 64), [(x_center, y_center)], sigma).to('cuda')  # shape: (64, 64)
            # 扩大高值区域
            heatmap = heatmap ** 0.01
            # heatmap: (64, 64) 广播
            heatmap = heatmap.unsqueeze(-1).expand(-1, -1, 768)  # → (64, 64, 768)
            # 在最前面加一个 batch 维度
            heatmap = heatmap.unsqueeze(0)        # → (1, 64, 64, 768)

            # 加入 box_tensor 特征图
            box_tensor += heatmap  # (1, 64, 64, 768)

            # 计算 block 处插入的索引
            first = 0
            quarter = len(self.blocks) // 4
            half = len(self.blocks) // 2
            three_quarters = 3 * len(self.blocks) // 4

            for i, blk in enumerate(self.blocks):
                if i == 5:  # 第6层（索引从0开始）
                    self.shallow_feature = x  # 形状: (B, H, W, C) = (1, 64, 64, 768)
                    self.anchor_map = self.anchor(self.shallow_feature)
                    if self.anchor_map == None:
                        print("Edge Anchor Wrong!")
                
                if i in [first, quarter, half, three_quarters]:
                    print("i:", i)
                    x = x + box_tensor  # 在1/4, 2/4, 3/4处添加box_tensor
                    
                x = blk(x)
                
            self.deep_feature = x
                
        elif input_points==None and input_box==None and feature_map==None:
            # print("SAM!")

            for i, blk in enumerate(self.blocks):            
                if i == 5:  # 第6层（索引从0开始）
                    self.shallow_feature = x  # 形状: (B, H, W, C) = (1, 64, 64, 768)
                    self.anchor_map = self.anchor(self.shallow_feature)
                    if self.anchor_map == None:
                        print("Edge Anchor Wrong!")
                        
                x = blk(x)
                
            self.deep_feature = x
            
        else:
            raise ValueError("Wrong！")

        # print("x in image_encoder before permute:", x.shape)

        #调整维度后再送入neck层
        temp_x=x.permute(0, 3, 1, 2)
        # print("x in image_encoder after permute:", x.shape)
        x = self.neck(temp_x)
        # print("x in image_encoder at last:", x.shape)
        
        if input_box != None:
            x1, y1, x2, y2 = input_box
            is_good = self.anchorChecker.is_high_quality(anchor_map=self.anchor_map, box=(x1, y1, x2, y2))
            
            if is_good:
                loss_align = boundary_consistency_loss(
                    self.shallow_feature,
                    self.deep_feature,
                    self.anchor_map
                )
                print("High anchor!")
            else:
                # 设置一个无效值
                loss_align = torch.tensor(0.0, device=self.shallow_feature.device, requires_grad=True)
                print("Loss_align is invalide!")
        else:
            # 设置一个无效值
            loss_align = torch.tensor(0.0, device=self.shallow_feature.device, requires_grad=True)
            print("Anchor!")
            
        return x, loss_align

#规范+注意力+（残差）-规范+MLP
class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        # self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
    
    # 位置增强
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        # 768/12=64
        head_dim = dim // num_heads
        # 缩放因子为8
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

        self.Attn = None  # 保存注意力分数
    
    # 位置增强
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        
        self.Attn=attn
        # sprint("Attn.shape:", self.Attn.shape)
        
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        print("x in patchEmbed:", x.shape)
        x = x.permute(0, 2, 3, 1)
        return x
