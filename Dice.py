import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    gt_target = ((gt_mask == 0) | (gt_mask == 128)).astype(np.float32)  # Black (0) or Gray (128) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target)
    
    return (2. * intersection + smooth) / (union + smooth)
    

def validate_and_pair_files(pred_dir, gt_dir):

    paired_files = []
    
    # ground truth
    gt_files = {}
    for f in os.listdir(gt_dir):
        if not f.lower().endswith(('.png', '.jpg')):
            continue
        base_name = os.path.splitext(f)[0]
        if base_name in gt_files:
            print(f"Warn：repetitive ground truth {f} and {gt_files[base_name]}")
        gt_files[base_name] = f

    for pred_file in os.listdir(pred_dir):
        if not pred_file.lower().endswith(('.png', '.jpg')):
            continue
        
        base_name = os.path.splitext(pred_file)[0]
        gt_file = gt_files.get(base_name)
        
        if not gt_file:
            print(f"Warn：Not found {base_name}，skip {pred_file}")
            continue
            
        paired_files.append({
            'pred_path': os.path.join(pred_dir, pred_file),
            'gt_path': os.path.join(gt_dir, gt_file),
            'file_id': base_name
        })
    
    print(f"Success {len(paired_files)} ")
    return paired_files
def validate_and_pair_files_ISIC(pred_dir, gt_dir):

    paired_files = []
    
    gt_files = {}
    for f in os.listdir(gt_dir):
        if not f.lower().endswith(('.png', '.jpg')):
            continue
        base_name = os.path.splitext(f)[0]  # example：image1_segmentation
        gt_files[base_name] = f
    
    for pred_file in os.listdir(pred_dir):
        if not pred_file.lower().endswith(('.png', '.jpg')):
            continue
        
        pred_base = os.path.splitext(pred_file)[0]  # example：image1
        gt_base = f"{pred_base}_segmentation"       # example：image1_segmentation
        
        gt_file = gt_files.get(gt_base)
        
        if not gt_file:
            print(f"Warn：Not found {gt_base}，skip {pred_file}")
            continue
            
        paired_files.append({
            'pred_path': os.path.join(pred_dir, pred_file),
            'gt_path': os.path.join(gt_dir, gt_file),
            'file_id': pred_base 
        })
    
    print(f"Success {len(paired_files)}")
    return paired_files

def load_and_preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossibly read mask file：{mask_path}")
    return mask

def compute_dice_scores(paired_files):
    results = []
    
    for pair in tqdm(paired_files, desc="Compute Dice"):
        try:
            gt_mask = load_and_preprocess_mask(pair['gt_path'])
            
            pred_mask = load_and_preprocess_mask(pair['pred_path'])
            
            if gt_mask.shape != pred_mask.shape:
                raise ValueError(f"Shape mismatch：GT {gt_mask.shape} vs Pred {pred_mask.shape}")
                
            dice = calculate_dice(gt_mask, pred_mask) # others
            # dice = calculate_dice_black_target(gt_mask, pred_mask) # REFUGE_CUP
            # dice = calculate_dice_black_gray_target(gt_mask, pred_mask) # REFUGE_DISC
            
            
            results.append({
                'file_id': pair['file_id'],
                'dice': dice,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"Processing {pair['file_id']} wrong：{str(e)}")
            results.append({
                'file_id': pair['file_id'],
                'dice': np.nan,
                'status': f'error: {str(e)}'
            })
    
    return results

def save_results(results, output_dir):
    df = pd.DataFrame(results)
    
    valid_df = df[df['status'] == 'success']
    mean_dice = valid_df['dice'].mean()
    
    df = pd.concat([
        df,
        pd.DataFrame([{
            'file_id': 'MEAN',
            'dice': mean_dice,
            'status': 'success'
        }])
    ])
    
    output_path = os.path.join(output_dir, 'dice_scores.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults has been saved in：{output_path}")
    print(f"Mean Dice：{mean_dice:.4f}")

if __name__ == "__main__":
    pred_dir = "dlcv/sam/Kvasir/box2/predictions" # predictions path
    gt_dir = "data/Kvasir-SEG/test/masks" 
    output_dir = "dlcv/sam/Kvasir/box2" 
    
    paired_files = validate_and_pair_files(pred_dir, gt_dir)
    
    results = compute_dice_scores(paired_files)
    save_results(results, output_dir)
