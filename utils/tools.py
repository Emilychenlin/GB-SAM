import numpy as np
import torch
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from box import Box
import csv
import copy

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

def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points) 
        print("point!")
        # prompts = get_point_prompts_seg(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
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
            sampled_points = np.flip(sampled_points, axis=1)  #  [x, y]
        
        n_points.append(sampled_points)

    return np.array(n_points) # (N, 2) or (1, 2)

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


# def k_means_sampling(mask, k):
#     points = np.argwhere(mask == 1) # [y, x]
#     points = np.flip(points, axis=1)

#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(points)
#     points = kmeans.cluster_centers_
#     return points


def get_point_prompt_max_dist(mask, num_points):
    n_points = []
    # for mask in masks:
    # mask_np = mask.cpu().numpy()
    mask_np=mask

    indices = np.argwhere(mask_np ==255 )
    random_index = np.random.choice(len(indices), 1)[0]

    first_point = [indices[random_index][1], indices[random_index][0]]
    new_points = get_multi_distance_points(first_point, mask_np, num_points - 1)
    n_points.append(new_points)

    return np.array(n_points)


def get_point_prompt_within_mask(mask, num_points, radius=50):
    n_points = []
    # for mask in masks:
    mask_np = mask
    print("mask.shape in get_point_prompt_within_mask:", mask_np.shape)
    y, x = np.where(mask_np == 255)  
    white_pixels = list(zip(y, x)) 
    
    if len(white_pixels) == 0:
        n_points.append(np.zeros((num_points, 2)))  
        print("返回空点！")
        # continue

    center_y, center_x = np.mean(y), np.mean(x)

    points = []
    points.append([center_y, center_x])

    for _ in range(num_points - 1):
        point = white_pixels[np.random.randint(0, len(white_pixels))]
        points.append(point)

    n_points.append(np.array(points))  
    
    return np.array(n_points)

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

        print("po_point_coords shape:", po_point_coords.shape)  # (num_points, 2)

        po_point_labels = torch.ones(po_point_coords.shape[0], dtype=torch.int, device=po_point_coords.device) 

        print(f"Final point_coords shape: {po_point_coords.shape}")  
        print(f"Final point_labels shape: {po_point_labels.shape}") 

        in_points = (po_point_coords, po_point_labels)
        prompts.append(in_points)

    return prompts

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
    
import matplotlib.pyplot as plt

def save_cam_overlay(img_np, heatmap, cam_dir, file_prefix):

    os.makedirs(cam_dir, exist_ok=True)
    
    # img_np = image_tensor.permute(1,2,0).cpu().numpy()
    heatmap_np = heatmap.cpu().numpy()
    
    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
    plt.axis('off')

    save_path = os.path.join(cam_dir, f"{file_prefix}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"CAM image saved to {save_path}")
    

# from torchsummary import summary

def freeze(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def momentum_update(student_model, teacher_model, momentum=0.99):
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        student_model.named_parameters(), teacher_model.named_parameters()
    ):
        if src_param.requires_grad:
            tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1 - momentum)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


def copy_model(model: torch.nn.Module):
    new_model = copy.deepcopy(model)
    freeze(new_model)
    return new_model


def create_csv(filename, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    if os.path.exists(filename):
        return 
    with open(filename, 'w') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head)
        csv_write.writeheader()


def write_csv(filename, csv_dict, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    with open(filename, 'a+') as csvfile:
        csv_write = csv.DictWriter(csvfile, fieldnames=csv_head, extrasaction='ignore')
        csv_write.writerow(csv_dict)


def check_grad(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def check_equal(model1: torch.nn.Module, model2: torch.nn.Module):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:
            if not torch.allclose(param1, param2):
                print(f"{name1} is different")
            else:
                print(f"same")
        else:
            print("The models have different structures")


# def check_model(model):
#     return summary(model, (3, 1024, 1024), batch_size=1, device="cuda")

def reduce_instances(bboxes, gt_masks, max_nums=50):
    bboxes_ = []
    gt_masks_ = []
    for bbox, gt_mask in zip(bboxes, gt_masks):
        idx = np.arange(bbox.shape[0])
        np.random.shuffle(idx)
        bboxes_.append(bbox[idx[:max_nums]])
        gt_masks_.append(gt_mask[idx[:max_nums]])

    bboxes = bboxes_
    gt_masks = gt_masks_
    return bboxes, gt_masks
