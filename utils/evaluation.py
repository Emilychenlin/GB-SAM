import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def calculate_dice(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate Dice coefficient for gt_mask and pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target region as 255 (white).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient.
    """
    # Ensure masks are binary for target regions
    gt_target = (gt_mask == 255).astype(np.float32)  # White (255) as target for gt_mask
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target for pred_mask
    
    # Calculate intersection and union
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target)
    
    return (2. * intersection + smooth) / (union + smooth)



def calculate_dice_black_target(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate Dice coefficient for gt_mask with black (0) as target region and 
    pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target region as 0 (black).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient.
    """
    # Ensure masks are binary for target regions
    gt_target = (gt_mask == 0).astype(np.float32)  # Black (0) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target)
    
    return (2. * intersection + smooth) / (union + smooth)

def calculate_dice_black_gray_target(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate Dice coefficient for gt_mask with black (0) and gray (128) as target regions 
    and pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target regions as 0 (black) and 128 (gray).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient.
    """
    # Ensure masks are binary for target regions
    # print("真实掩码唯一值:", np.unique(gt_mask))  # 应包含0（目标区域）
    # print("预测掩码唯一值:", np.unique(pred_mask))  # 应包含255（目标区域）

    gt_target = ((gt_mask == 0) | (gt_mask == 128)).astype(np.float32)  # Black (0) or Gray (128) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target)
    
    return (2. * intersection + smooth) / (union + smooth)

def calculate_miou(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate mean Intersection over Union (mIoU) for gt_mask and pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target region as 255 (white).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: mean Intersection over Union (mIoU).
    """
    # Ensure masks are binary for target regions
    gt_target = (gt_mask == 255).astype(np.float32)  # White (255) as target for gt_mask
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target for pred_mask
    
    # Calculate intersection and union for target class (white)
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target) - intersection
    
    # Calculate IoU for target class
    iou_target = (intersection + smooth) / (union + smooth)
    
    # For background class (non-white, 0), invert the masks
    gt_background = (gt_mask == 0).astype(np.float32)
    pred_background = (pred_mask == 0).astype(np.float32)
    
    # Calculate intersection and union for background class
    intersection_bg = np.sum(gt_background * pred_background)
    union_bg = np.sum(gt_background) + np.sum(pred_background) - intersection_bg
    
    # Calculate IoU for background class
    iou_background = (intersection_bg + smooth) / (union_bg + smooth)
    
    # Calculate mean IoU (average of target and background IoU)
    return (iou_target + iou_background) / 2.0

def calculate_miou_black_target(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate mean Intersection over Union (mIoU) for gt_mask with black (0) as target region 
    and pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target region as 0 (black).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: mean Intersection over Union (mIoU).
    """
    # Ensure masks are binary for target regions
    gt_target = (gt_mask == 0).astype(np.float32)  # Black (0) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union for target class
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target) - intersection
    
    # Calculate IoU for target class
    iou_target = (intersection + smooth) / (union + smooth)
    
    # For background class (non-black in gt, non-white in pred)
    gt_background = (gt_mask != 0).astype(np.float32)  # Anything not black
    pred_background = (pred_mask != 255).astype(np.float32)  # Anything not white
    
    # Calculate intersection and union for background class
    intersection_bg = np.sum(gt_background * pred_background)
    union_bg = np.sum(gt_background) + np.sum(pred_background) - intersection_bg
    
    # Calculate IoU for background class
    iou_background = (intersection_bg + smooth) / (union_bg + smooth)
    
    # Calculate mean IoU
    return (iou_target + iou_background) / 2.0

def calculate_miou_black_gray_target(gt_mask, pred_mask, smooth=1e-8):
    """
    Calculate mean Intersection over Union (mIoU) for gt_mask with black (0) and gray (128) 
    as target regions and pred_mask with white (255) as target region.
    
    Args:
        gt_mask (np.ndarray): Ground truth mask with target regions as 0 (black) and 128 (gray).
        pred_mask (np.ndarray): Predicted mask with target region as 255 (white).
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: mean Intersection over Union (mIoU).
    """
    # Ensure masks are binary for target regions
    gt_target = ((gt_mask == 0) | (gt_mask == 128)).astype(np.float32)  # Black (0) or Gray (128) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union for target class
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target) - intersection
    
    # Calculate IoU for target class
    iou_target = (intersection + smooth) / (union + smooth)
    
    # For background class (neither black nor gray in gt, non-white in pred)
    gt_background = (~((gt_mask == 0) | (gt_mask == 128))).astype(np.float32)  # Neither black nor gray
    pred_background = (pred_mask != 255).astype(np.float32)  # Anything not white
    
    # Calculate intersection and union for background class
    intersection_bg = np.sum(gt_background * pred_background)
    union_bg = np.sum(gt_background) + np.sum(pred_background) - intersection_bg
    
    # Calculate IoU for background class
    iou_background = (intersection_bg + smooth) / (union_bg + smooth)
    
    # Calculate mean IoU
    return (iou_target + iou_background) / 2.0