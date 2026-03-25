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
    prompts = []
    for mask in gt_masks:
        print("mask.shape in get_point_prompts:", mask.shape)
        po_points = uniform_sampling(mask, num_points)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)
        print("po_point_coords shape:", po_point_coords.shape)  # (num_points, 2)

        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device) 
        print(f"Final point_coords shape: {po_point_coords.shape}")  
        print(f"Final point_labels shape: {po_point_labels.shape}")  

        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)

    return prompts
