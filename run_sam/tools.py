import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from scipy.ndimage import center_of_mass
from PIL import Image

def generate_bounding_boxes_from_mask(mask):
    """
    从二值掩码图像生成包围白色区域的边界框。
    :param mask: 输入的二值图像，白色区域为255，其他为0。
    :return: 包围白色区域的边界框列表，每个框格式为 [x_min, y_min, x_max, y_max]。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255
    
    # 找到白色区域的坐标
    # 这里我们假设白色区域是255，黑色区域是0
    y_indices, x_indices = np.where(mask == 255)  # 获取白色区域的所有坐标

    if len(x_indices) == 0 or len(y_indices) == 0:
        return []  # 没有白色区域时返回空列表

    # 获取白色区域的最小和最大坐标，形成边界框
    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)

    bounding_box = [[x_min, y_min, x_max, y_max]]

    return bounding_box

def visualize_and_save_masks(masks, output_dir, img_name, expected_size=None):
    """
    保存单通道二值掩码为 PNG 图像，并确保不改变原图大小

    Args:
        masks (np.ndarray): 二值掩码，形状为 (1, H, W)
        output_dir (str): 掩码保存目录
        file_name_prefix (str): 输出文件名前缀
        expected_size (tuple): 可选，期望的图像大小 (W, H)，如与原图一致时可校验
    """
    import os
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    mask = masks[0]  # (H, W)

    if expected_size is not None:
        assert mask.shape[::-1] == expected_size, f"掩码尺寸 {mask.shape[::-1]} 与原图尺寸 {expected_size} 不一致"

    # 保存时不改变尺寸
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')  # 'L' 表示灰度图

    output_path = os.path.join(output_dir, f"{img_name}.png")
    mask_img.save(output_path)
    
def get_point_prompts_seg(gt_masks, num_points):
    """
    生成前景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
    """
    prompts = []
    for mask in gt_masks:
        # mask = (mask > 0).to(torch.uint8)  # 归一化成 0/1
        
        # **获取前景点**
        # po_points = get_point_prompt_within_mask([mask.cpu().numpy()], num_points)
        print("mask.shape in get_point_prompts:", mask.shape)
        po_points = uniform_sampling(mask, num_points)
        # po_points = sample_black_gray_points(mask, debug=True)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        # **转换为 PyTorch tensor**
        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

        print("po_point_coords shape:", po_point_coords.shape)  # 预期 (num_points, 2)

        # **创建对应的 labels**
        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device)  # 前景点标签为 1

        print(f"Final point_coords shape: {po_point_coords.shape}")  # 应该是 (num_points, 2)
        print(f"Final point_labels shape: {po_point_labels.shape}")  # 应该是 (num_points,)

        # **打包成 tuple**
        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)
        # point_color = (0, 255, 255)  # 采用红色标注点
        # radius = 5  # 设置点的大小
        # print("point_coords in use:", po_point_coords.shape)
        # mask=mask
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为三通道 RGB 图像
        # print("masks before circle:", mask.shape)
        # for (x, y) in po_point_coords:
        #     cv2.circle(mask, (int(x), int(y)), radius, point_color, -1)  # -1 表示填充圆
        # print("masks after circle:", mask.shape)
        # image_pil = Image.fromarray(mask)
        # image_pil.save('output_in_get_point_prompts.png')
        # exit()

    return prompts

def uniform_sampling(mask, N):
    """
    从mask中均匀采样N个点，确保返回形状为 (N, 2)。当N=1时，取白色区域中心点。
    """
    n_points = []
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy().astype(np.uint8)

    print("打印掩码中的唯一值：", np.unique(mask))  # 打印掩码中的唯一值
    if np.sum(mask == 255) == 0:
        indices = np.argwhere(mask == 0)  # 采样背景点
    elif np.sum(mask == 0) == 0:
        indices = np.argwhere(mask == 255)  # 采样前景点
    else:
        indices = np.argwhere(mask == 255)  # 采样前景点

    # **防止空数组**
    if len(indices) == 0:
        print("Warning: No valid points found in mask, skipping.")
        n_points.append(np.zeros((N, 2)))  # 避免后续代码崩溃
    else:
        if N == 1:
            # 当N=1时，计算白色区域的中心点
            center_y = np.mean(indices[:, 0])  # y坐标均值
            center_x = np.mean(indices[:, 1])  # x坐标均值
            sampled_points = np.array([[center_x, center_y]])  # 形状为 (1, 2)
        else:
            # 其他情况下，随机均匀采样
            sampled_indices = np.random.choice(len(indices), N, replace=True)
            sampled_points = indices[sampled_indices]  # (N, 2), 顺序是 [y, x]
            sampled_points = np.flip(sampled_points, axis=1)  # 变为 [x, y]
        
        n_points.append(sampled_points)

    return np.array(n_points) # 返回 (N, 2) 或 (1, 2)
