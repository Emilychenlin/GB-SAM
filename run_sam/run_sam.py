import os
import numpy as np
import torch
from segment_anything.predictor import SamPredictor
from configs.config import cfg
# from dataset.ISIC import load_datasets
from dataset.Kvasir import load_dataset
from tools import get_point_prompts_seg,generate_bounding_boxes_from_mask, visualize_and_save_masks
from segment_anything.build_sam import sam_model_registry
import cv2
from Dice import validate_and_pair_files, compute_dice_scores, save_results

# from overlay import overlay_bounding_boxes, overlay_heatmap, compute_grad_cam, visualize_gradients_box, visualize_gradients_point, process_grad_cam, visualize_activation

def save_grad_cam_image(output_dir, img_name, layer_idx, overlayed_img):
    # 创建以 batch_idx 命名的文件夹
    batch_dir = os.path.join(output_dir, "heatmap", img_name)
    os.makedirs(batch_dir, exist_ok=True)

    # 构建文件路径并保存图像
    file_path = os.path.join(batch_dir, f"layer_{layer_idx}.png")
    cv2.imwrite(file_path, overlayed_img)

def safe_cv2_circle(image, center, radius, color, thickness):
    """安全封装OpenCV绘图函数"""
    # 转换数据类型和内存布局
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy().astype(np.uint8)
    else:
        image_np = image.astype(np.uint8)
    
    image_np = np.ascontiguousarray(image_np)
    
    # 执行绘图操作
    cv2.circle(image_np, tuple(center), radius, color, thickness)
    
    # 如果需要返回张量
    return torch.from_numpy(image_np).to(image.device) if isinstance(image, torch.Tensor) else image_np

def main(cfg, ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    # 修改数据集文件夹路径
    data_path='/home/chenlin/Desktop/tent_924/data/Kvasir-SEG/test/images' 
    mask_path='/home/chenlin/Desktop/tent_924/data/Kvasir-SEG/test/masks'
    train_dataloader = load_dataset(data_path, mask_path)
    
    sam_checkpoint = os.path.abspath("sam_vit_b_01ec64.pth") # 修改为我们自己微调过后的权重文件
    model_type = cfg.model.type
    print(model_type)
    print(sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    # sam.to(torch.bfloat16)

    print("加载SAM模型完成！")

    predictor = SamPredictor(sam)
    print("加载预测模型")

    # 修改输出文件夹路径
    output_dir = 'results/sam/box/Kvasir'
    os.makedirs(output_dir, exist_ok=True)
    print("输出文件夹路径", output_dir)

    with torch.no_grad():
        for batch_idx, (images, masks, img_ids) in enumerate(train_dataloader):
            if batch_idx >= 0:
                images = images.to(device)
                masks = masks.to(device)
                img_name = img_ids[0]

                image = images[0].cpu().numpy().transpose(1, 2, 0)
                if image.dtype == np.float32:
                    image = (image * 255).astype(np.uint8)
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                masks=masks.cpu().numpy()
                masks = (masks * 255).astype(np.uint8)  # 将掩码值从 [0, 1] 转换为 [0, 255]
                
                # ########################################################################################点（默认随机选取三个点！！！！！！！！！！！）
                # print("num_points:", cfg.num_points)
                # print("mask before get_point:", masks.shape)
                # prompts = get_point_prompts_seg(masks, cfg.num_points)
                # point_coords, point_labels = prompts[0]

                # ## prompt encoder in image encoder时候需要转换为整数
                # # point_coords = point_coords.to(torch.int)

                # print("point_coords:", point_coords)
                # print("point_coords.shape:", point_coords.shape)
                # print("image:", image.shape)

                # # point_coords = point_coords.cpu().numpy().reshape(1, 2)
                # point_coords = point_coords.cpu().numpy()
                # point_labels = point_labels.cpu().numpy()

                # predictor.set_image(
                #     image = image,
                # )

                # if point_coords.ndim != 2 or point_coords.shape[-1] != 2:
                #     raise ValueError(f"point_coords must have shape (N, 2), but got {point_coords.shape}")
                
                # pred_masks, scores, logits = predictor.predict(
                #     point_coords=point_coords,
                #     point_labels=point_labels,
                #     multimask_output=False
                # )


                # ##############################################################################框
                
                prompts = generate_bounding_boxes_from_mask(masks)
                box_coords = np.array(prompts[0])  # 确保它是一个 (4,) 数组
                box_coords = box_coords.reshape(1, 4)  # 转换成 (1, 4) 的二维数组
                print("box_coords shape:", box_coords.shape)
                print("box_coords:", box_coords)

                # 将prompt和imput一起输入image_encoder
                predictor.set_image(
                    image = image,
                )

                # 进行预测
                pred_masks, scores, logits = predictor.predict(
                    box=box_coords,
                    multimask_output=False
                )
                print("box_coords:", box_coords)
                
                # 输出预测结果
                output_dir_masks = os.path.join(output_dir, "predictions")
                print("pred_masks.shape:", pred_masks.shape)

                visualize_and_save_masks(
                    masks=pred_masks,
                    output_dir=output_dir_masks,
                    img_name=f"{img_name}"
                )

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
    paired_files = validate_and_pair_files(output_dir_masks, mask_path)
    results = compute_dice_scores(paired_files)
    save_results(results, output_dir)


if __name__ == "__main__":
    main(cfg)
