import os
import torch
import lightning as L
from box import Box
from torch.utils.data import DataLoader
from utils.sample_utils import get_point_prompts, get_point_prompts_seg
from utils.tools import write_csv
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PIL import Image, ImageDraw

def calculate_dice_torch(gt_mask: torch.Tensor, pred_mask: torch.Tensor, smooth=1e-8):

    gt_mask = gt_mask.float()
    pred_mask = pred_mask.float()

    intersection = torch.sum(gt_mask * pred_mask, dim=(-2, -1))
    union = torch.sum(gt_mask, dim=(-2, -1)) + torch.sum(pred_mask, dim=(-2, -1))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_prob = torch.sigmoid(pred_mask)
    pred_bin = (pred_prob >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


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

import torchvision.transforms.functional as TF

def save_image_with_boxes(img_tensor, boxes, save_path="vis.png"):
    """
    img_tensor: (1, 3, H, W) tensor
    boxes: List of boxes, each box is [x1, y1, x2, y2]
    """

    img_tensor = img_tensor.squeeze(0).cpu()
    img = TF.to_pil_image(img_tensor)

    draw = ImageDraw.Draw(img)

    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        box = [int(x) for x in box]
        draw.rectangle(box, outline="red", width=3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


import os
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import torch

import matplotlib.pyplot as plt

def save_image_with_points(img_tensor, coords, labels, save_path):

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)  # [3, H, W]

    img = TF.to_pil_image(img_tensor.cpu())

    draw = ImageDraw.Draw(img)

    for i in range(len(coords)):
        x, y = coords[i]
        x, y = float(x), float(y)
        r = 5  

        if labels[i] == 1:
            fill_color = "green"
        else:
            fill_color = "red"

        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill_color, outline=None)

    img.save(save_path)


def save_image_with_boxes(img_tensor, box_list, save_path):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img = TF.to_pil_image(img_tensor.cpu())
    draw = ImageDraw.Draw(img)
    for box in box_list:
        x1, y1, x2, y2 = map(float, box.tolist())
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    img.save(save_path)
