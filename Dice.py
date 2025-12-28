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
    # print("真实掩码唯一值:", np.unique(gt_mask))  # 应包含0（目标区域）
    # print("预测掩码唯一值:", np.unique(pred_mask))  # 应包含255（目标区域）

    gt_target = ((gt_mask == 0) | (gt_mask == 128)).astype(np.float32)  # Black (0) or Gray (128) as target
    pred_target = (pred_mask == 255).astype(np.float32)  # White (255) as target
    
    # Calculate intersection and union
    intersection = np.sum(gt_target * pred_target)
    union = np.sum(gt_target) + np.sum(pred_target)
    
    return (2. * intersection + smooth) / (union + smooth)
    

def validate_and_pair_files(pred_dir, gt_dir):
    """
    改进版文件配对逻辑：通过文件名（无扩展名）匹配
    """
    paired_files = []
    
    # 构建ground truth文件的基名索引
    gt_files = {}
    for f in os.listdir(gt_dir):
        if not f.lower().endswith(('.png', '.jpg')):
            continue
        base_name = os.path.splitext(f)[0]
        if base_name in gt_files:
            print(f"警告：发现重复的ground truth文件 {f} 和 {gt_files[base_name]}")
        gt_files[base_name] = f
    
    # 配对预测文件
    for pred_file in os.listdir(pred_dir):
        if not pred_file.lower().endswith(('.png', '.jpg')):
            continue
        
        base_name = os.path.splitext(pred_file)[0]
        gt_file = gt_files.get(base_name)
        
        if not gt_file:
            print(f"警告：未找到匹配的真值文件 {base_name}，已跳过 {pred_file}")
            continue
            
        paired_files.append({
            'pred_path': os.path.join(pred_dir, pred_file),
            'gt_path': os.path.join(gt_dir, gt_file),
            'file_id': base_name
        })
    
    print(f"成功配对 {len(paired_files)} 对文件")
    return paired_files
def validate_and_pair_files_ISIC(pred_dir, gt_dir):
    """
    改进版文件配对逻辑：GT文件名 = 预测文件名 + "_segmentation"
    """
    paired_files = []
    
    # 构建ground truth文件的基名索引（自动去除扩展名）
    gt_files = {}
    for f in os.listdir(gt_dir):
        if not f.lower().endswith(('.png', '.jpg')):
            continue
        base_name = os.path.splitext(f)[0]  # 例如：image1_segmentation
        gt_files[base_name] = f
    
    # 配对预测文件
    for pred_file in os.listdir(pred_dir):
        if not pred_file.lower().endswith(('.png', '.jpg')):
            continue
        
        # 生成对应的GT基名
        pred_base = os.path.splitext(pred_file)[0]  # 例如：image1
        gt_base = f"{pred_base}_segmentation"       # 例如：image1_segmentation
        
        # 查找匹配的gt文件
        gt_file = gt_files.get(gt_base)
        
        if not gt_file:
            print(f"警告：未找到匹配的真值文件 {gt_base}，已跳过 {pred_file}")
            continue
            
        paired_files.append({
            'pred_path': os.path.join(pred_dir, pred_file),
            'gt_path': os.path.join(gt_dir, gt_file),
            'file_id': pred_base  # 保持原始预测文件名作为标识
        })
    
    print(f"成功配对 {len(paired_files)} 对文件（匹配模式：预测文件名 + '_segmentation'）")
    return paired_files

def load_and_preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩码文件：{mask_path}")
    return mask

def compute_dice_scores(paired_files):
    results = []
    
    for pair in tqdm(paired_files, desc="计算Dice分数"):
        try:
            gt_mask = load_and_preprocess_mask(pair['gt_path'])
            
            pred_mask = load_and_preprocess_mask(pair['pred_path'])
            
            if gt_mask.shape != pred_mask.shape:
                raise ValueError(f"形状不匹配：GT {gt_mask.shape} vs Pred {pred_mask.shape}")
                
            dice = calculate_dice(gt_mask, pred_mask) # 其他数据集
            # dice = calculate_dice_black_target(gt_mask, pred_mask) # REFUGE_CUP数据集
            # dice = calculate_dice_black_gray_target(gt_mask, pred_mask) # REFUGE_DISC数据集
            
            
            results.append({
                'file_id': pair['file_id'],
                'dice': dice,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"处理文件 {pair['file_id']} 时出错：{str(e)}")
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
    print(f"\n结果已保存至：{output_path}")
    print(f"平均Dice系数：{mean_dice:.4f}")

if __name__ == "__main__":
    pred_dir = "dlcv/sam/Kvasir/box2/predictions" # predictions的路径
    gt_dir = "data/Kvasir-SEG/test/masks" # 真实值的路径
    output_dir = "dlcv/sam/Kvasir/box2" #输出文件夹路径（存放每张图片的值以及最后整个数据集上的均值）
    
    # #ISIC数据集使用
    # paired_files = validate_and_pair_files_ISIC(pred_dir, gt_dir)
    #其它数据集使用
    paired_files = validate_and_pair_files(pred_dir, gt_dir)
    
    results = compute_dice_scores(paired_files)
    save_results(results, output_dir)
