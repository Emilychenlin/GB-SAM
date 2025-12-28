import numpy as np
import torch
from sklearn.cluster import KMeans
import cv2


def uniform_sampling(masks, N=3):
    n_points = []
    for mask in masks:
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()

        print("masks.shape:", masks.shape)
        print("masks.max:", masks.max())

        if mask.max() == 1:
            indices = np.argwhere(mask == 1)
        else:
            indices = np.argwhere(mask == 255) # [y, x]
        sampled_indices = np.random.choice(len(indices), N, replace=True)
        sampled_points = np.flip(indices[sampled_indices], axis=1)
        n_points.append(sampled_points.tolist())

    return n_points

import numpy as np
import cv2

# def uniform_sampling(masks, N=3, max_region_ratio=2.0):
#     """
#     从连通区域中采样点：
#     - 优先选择面积较大且面积相差不大的区域；
#     - 每个区域至少一个点；
#     - 总共返回 N 个点。

#     参数:
#         masks: List of binary masks (H, W), 0/1 or 0/255
#         N: 总采样点数
#         max_region_ratio: 面积容差系数（最大面积 / 当前区域面积 <= ratio 才保留）
#     返回:
#         n_points: List of N (x, y) 坐标点列表
#     """
#     n_points = []

#     for mask in masks:
#         if not isinstance(mask, np.ndarray):
#             mask = mask.cpu().numpy()

#         # 归一化为 0/1
#         mask = (mask > 0).astype(np.uint8)

#         # 连通区域分析
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

#         # 获取各区域面积（排除背景，第0个）
#         areas = stats[1:, cv2.CC_STAT_AREA]
#         sorted_indices = np.argsort(areas)[::-1]  # 从大到小排序
#         largest_area = areas[sorted_indices[0]] if len(areas) > 0 else 0

#         # 过滤掉过小的噪声区域
#         valid_region_ids = []
#         for idx in sorted_indices:
#             if largest_area / (areas[idx] + 1e-6) <= max_region_ratio:
#                 valid_region_ids.append(idx + 1)  # +1 因为 stats 不包含背景 label=0
#             if len(valid_region_ids) >= N:  # 最多取 N 个区域
#                 break

#         points = []

#         # 每个有效区域采样 1 个点
#         for region_id in valid_region_ids:
#             region_coords = np.argwhere(labels == region_id)
#             if len(region_coords) == 0:
#                 continue
#             sampled_idx = np.random.choice(len(region_coords))
#             pt = region_coords[sampled_idx][::-1]  # (y, x) -> (x, y)
#             points.append(pt.tolist())

#         # 不足 N 个点则从所有有效区域中补点
#         if len(points) < N:
#             fg_coords = np.argwhere(np.isin(labels, valid_region_ids))
#             if len(fg_coords) > 0:
#                 remaining = N - len(points)
#                 sampled_idx = np.random.choice(len(fg_coords), remaining, replace=True)
#                 extra_pts = np.flip(fg_coords[sampled_idx], axis=1)
#                 points.extend(extra_pts.tolist())

#         n_points.append(points)

#     return n_points

# def uniform_sampling(mask, N):
#     """
#     从mask中均匀采样N个点，确保返回形状为 (N, 2)。当N=1时，取白色区域中心点。
#     """
#     n_points = []
#     if not isinstance(mask, np.ndarray):
#         mask = mask.cpu().numpy().astype(np.uint8)

#     print("打印掩码中的唯一值：", np.unique(mask))  # 打印掩码中的唯一值
#     if np.sum(mask == 255) == 0:
#         indices = np.argwhere(mask == 0)  # 采样背景点
#     elif np.sum(mask == 0) == 0:
#         indices = np.argwhere(mask == 255)  # 采样前景点
#     else:
#         indices = np.argwhere(mask == 255)  # 采样前景点

#     # **防止空数组**
#     if len(indices) == 0:
#         print("Warning: No valid points found in mask, skipping.")
#         n_points.append(np.zeros((N, 2)))  # 避免后续代码崩溃
#     else:
#         if N == 1:
#             # 当N=1时，计算白色区域的中心点
#             center_y = np.mean(indices[:, 0])  # y坐标均值
#             center_x = np.mean(indices[:, 1])  # x坐标均值
#             sampled_points = np.array([[center_x, center_y]])  # 形状为 (1, 2)
#         else:
#             # 其他情况下，随机均匀采样
#             sampled_indices = np.random.choice(len(indices), N, replace=True)
#             sampled_points = indices[sampled_indices]  # (N, 2), 顺序是 [y, x]
#             sampled_points = np.flip(sampled_points, axis=1)  # 变为 [x, y]
        
#         n_points.append(sampled_points)

#     return np.array(n_points) # 返回 (N, 2) 或 (1, 2)


def get_multi_distance_points(input_point, mask, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[1], input_point[0]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_distance_point(new_points[:i + 1, :], mask)

    new_points = swap_xy(new_points)
    return new_points


def get_next_distance_point(input_points, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    input_points = np.array(input_points)

    indices = np.argwhere(mask == True)
    for x, y in indices:
        # print(x,y,input_points)
        distance = np.sum(np.sqrt((x - input_points[:, 0]) ** 2 + (y - input_points[:, 1]) ** 2))
        if max_distance < distance:
            max_distance_point = [x, y]
            max_distance = distance
    return max_distance_point


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points


def k_means_sampling(mask, k):
    points = np.argwhere(mask == 1) # [y, x]
    points = np.flip(points, axis=1)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    points = kmeans.cluster_centers_
    return points


def get_point_prompt_max_dist(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()

        indices = np.argwhere(mask_np > 0)
        random_index = np.random.choice(len(indices), 1)[0]

        first_point = [indices[random_index][1], indices[random_index][0]]
        new_points = get_multi_distance_points(first_point, mask_np, num_points - 1)
        n_points.append(new_points)

    return n_points


def get_point_prompt_kmeans(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()
        points = k_means_sampling(mask_np, num_points)
        n_points.append(points.astype(int))
    return n_points


def get_point_prompts(gt_masks, num_points):
    prompts = []
    for mask in gt_masks:
        po_points = uniform_sampling(mask, num_points)
        # na_points = uniform_sampling((~mask.to(bool)).to(float), num_points)
        po_point_coords = torch.tensor(po_points, device=mask.device)
        # na_point_coords = torch.tensor(na_points, device=mask.device)
        # point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
        po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=po_point_coords.device)
        # na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=na_point_coords.device)
        # point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)
    return prompts

def get_point_prompts_seg(gt_masks, num_points):
    """
    生成前景采样点，并返回 point_coords 和 point_labels，确保 shape 为 (N, 2)
    """
    prompts = []
    for mask in gt_masks:
        print("mask.shape in get_point_prompts:", mask.shape)
        po_points = uniform_sampling(mask, num_points)
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

    return prompts