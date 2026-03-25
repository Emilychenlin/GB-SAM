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

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    if mask.dtype != np.uint8:
        mask = (mask > 127).astype(np.uint8) * 255

    y_indices, x_indices = np.where(mask == 255) 

    if len(x_indices) == 0 or len(y_indices) == 0:
        return []  

    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)

    bounding_box = [[x_min, y_min, x_max, y_max]]

    return bounding_box

def visualize_and_save_masks(masks, output_dir, img_name, expected_size=None):

    import os
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    mask = masks[0]  # (H, W)

    if expected_size is not None:
        assert mask.shape[::-1] == expected_size, f"掩码尺寸 {mask.shape[::-1]} 与原图尺寸 {expected_size} 不一致"

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L') 

    output_path = os.path.join(output_dir, f"{img_name}.png")
    mask_img.save(output_path)
    
def get_point_prompts_seg(gt_masks, num_points):

    prompts = []
    for mask in gt_masks:
        # mask = (mask > 0).to(torch.uint8) 
 
        # po_points = get_point_prompt_within_mask([mask.cpu().numpy()], num_points)
        print("mask.shape in get_point_prompts:", mask.shape)
        po_points = uniform_sampling(mask, num_points)
        # po_points = sample_black_gray_points(mask, debug=True)
        print("po_points.shape:", po_points.shape)  # (1, num_points, 2)

        po_point_coords = torch.tensor(po_points, dtype=torch.float32, device=mask.device).squeeze(0)  # (num_points, 2)

        print("po_point_coords shape:", po_point_coords.shape)  


        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device) 

        print(f"Final point_coords shape: {po_point_coords.shape}")  # (num_points, 2)
        print(f"Final point_labels shape: {po_point_labels.shape}")  # (num_points,)

        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)

    return prompts

def uniform_sampling(mask, N):

    n_points = []
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy().astype(np.uint8)

    print("Print the unique value in the mask：", np.unique(mask)) 
    if np.sum(mask == 255) == 0:
        indices = np.argwhere(mask == 0) 
    elif np.sum(mask == 0) == 0:
        indices = np.argwhere(mask == 255)  
    else:
        indices = np.argwhere(mask == 255)

    if len(indices) == 0:
        print("Warning: No valid points found in mask, skipping.")
        n_points.append(np.zeros((N, 2))) 
    else:
        if N == 1:

            center_y = np.mean(indices[:, 0])  
            center_x = np.mean(indices[:, 1]) 
            sampled_points = np.array([[center_x, center_y]])  # (1, 2)
        else:
            sampled_indices = np.random.choice(len(indices), N, replace=True)
            sampled_points = indices[sampled_indices]  # (N, 2),  [y, x]
            sampled_points = np.flip(sampled_points, axis=1)  # [x, y]
        
        n_points.append(sampled_points)

    return np.array(n_points) 
