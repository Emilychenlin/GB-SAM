import os
import yaml
import torch
import numpy as np
import lightning as L
from box import Box
from pytorch_lightning.loggers import CSVLogger
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

from configs.config import cfg
from datasets import call_load_dataset
from utils.tools import write_csv, visualize_and_save_masks, create_csv, get_prompts
from utils.evaluation import calculate_dice, calculate_miou, AverageMeter
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def collect_params_custom(predictor):
    """
    收集：
    1. image_encoder.blocks.11（ViT最后一层Block）
    2. image_encoder.heatmap_proj（高斯热力图投影层）
    3. image_encoder.anchor（Anchor模块）

    返回：
    - params: List[Parameter]
    - names: List[str]
    """
    target_names = [
        "image_encoder.blocks.11",
        # "image_encoder.heatmap_proj",
        # "image_encoder.anchor",
    ]

    params = []
    names = []

    for name, module in predictor.model.named_modules():
        for tn in target_names:
            if name == tn or name.startswith(tn + "."):
                for p_name, p in module.named_parameters(recurse=False):
                    params.append(p)
                    names.append(f"{name}.{p_name}")

    print(f" Collected {len(params)} params from specified modules.")
    return params, names

def configure_model_custom(predictor):
    """
    设置模型为部分可训练：
    只保留以下部分为 requires_grad=True：
    - image_encoder.blocks.11（最后一层Block）
    - image_encoder.heatmap_proj
    - image_encoder.anchor
    """
    # 整体冻结
    predictor.model.eval()
    predictor.model.requires_grad_(False)

    # 解冻目标模块
    target_names = [
        "image_encoder.blocks.11",
        # "image_encoder.heatmap_proj",
        # "image_encoder.anchor",
    ]

    for name, module in predictor.model.named_modules():
        for tn in target_names:
            if name == tn or name.startswith(tn + "."):
                for p in module.parameters(recurse=False):
                    p.requires_grad = True

    print(" Model configured: Only selected modules are set as trainable.")
    return predictor

def Adapt(fabric, optimizer, cfg, predictor, load_datasets, name=cfg.name, iters=0):
    ious = AverageMeter()
    f1_scores = AverageMeter()
    dice_scores = AverageMeter()

    test_dataloader = load_datasets(cfg, predictor.model.image_encoder.img_size)

    for iter, data in enumerate(test_dataloader):
        images, bboxes, gt_masks, img_names = data
        # print("images.shape:", images.shape)
        # print("box.shape:", bboxes.shape)

        img_name = img_names[0]
        num_images = images.size(0)

        # 创建保存目录
        pred_mask_dir = os.path.join(cfg.out_dir, "pred_masks")
        # matrices = os.path.join(cfg.out_dir, "matrices")
        os.makedirs(pred_mask_dir, exist_ok=True)

        # 将图像搬到 GPU
        images = images.to(fabric.device)

        # 获取 prompts
        prompts = get_prompts(cfg, bboxes, gt_masks)
        
        img = images[0].cpu()

        if img.dtype == torch.float32 or img.max() <= 1.0:
            img = img * 255.0

        # Step 2: 转为 numpy 并调整形状
        # 1. 如果值在 [0, 1] 或 [-1, 1]，先归一化到 [0, 255]
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        # 2. 转成 uint8 类型
        img = img.to(torch.uint8)
        # 3. 转成 numpy，CHW -> HWC
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        # print("img.shape:", img_np.shape)
        
        # print("prompts.shape:", prompts.shape)
        # print("prompts[0].shape:", prompts[0].shape)
        box_coords = np.array(prompts[0])  # 确保它是一个 (4,) 数组
        # print("box_coords.shape:", box_coords.shape)
        box_coords = box_coords.reshape(1, 4)  # 转换成 (1, 4) 的二维数组
        # print("box_coords shape:", box_coords.shape)
        # print("box_coords:", box_coords)
        
        loss_align = predictor.set_image(
            image = img_np,
            box = box_coords
        )

        # if point_coords.ndim != 2 or point_coords.shape[-1] != 2:
        #     raise ValueError(f"point_coords must have shape (N, 2), but got {point_coords.shape}")
        pred_masks, scores, logits = predictor.predict(
            box=box_coords,
            # point_coords=point_coords,
            # point_labels=point_labels,
            multimask_output=False
        )
        
        # print("pred_masks:", pred_masks.shape)
        # print("logits:", logits.shape)
        
        loss_align.backward()
        optimizer.step()
        # 更新梯度缓存
        optimizer.zero_grad()
        
        # 保存掩码图像
        visualize_and_save_masks(pred_masks, pred_mask_dir, img_name)

        # 处理 Ground Truth 掩码，保持一致性
        gt_masks = gt_masks[0]
        print("gt_masks.shape:", gt_masks.shape)
        if gt_masks.max() > 1:
            gt_bin = (gt_masks == 255).float()
        else:
            gt_bin = gt_masks.float()

        # 转为 numpy 格式，用于 Dice/mIoU 计算
        gt_np = gt_bin[0].cpu().numpy().astype(np.uint8) * 255  # -> (H, W), 取白色为目标区域
        # pred_np = pred_np[0] * 255  # (H, W)，转换为 0/255 格式
        # print("gt_np.shape:", gt_np.shape)
        # print("gt_np.max:", gt_np.max())
        # print("pred_masks.shape:", pred_masks.shape)
        # print("pred_masks.max:", pred_masks.max())
        # 从 (1, H, W) -> (H, W)
        pred_np = pred_masks[0]  # 去掉 batch 维
        # 转成 numpy 格式（如果还不是）
        if isinstance(pred_np, torch.Tensor):
            pred_np = pred_np.cpu().numpy()
        # 转成 uint8 二值图，背景 0，前景 255
        pred_np = pred_np.astype(np.uint8) * 255 

        # Dice 计算
        batch_dice = calculate_dice(gt_np, pred_np)
        # mIoU 计算
        batch_iou = calculate_miou(gt_np, pred_np)

        # F1-score 计算（F1 = 2 * precision * recall / (precision + recall)）
        # 这里我们用 Dice 近似 F1-score（因为二者在 binary segmentation 下等价）
        batch_f1 = batch_dice

        # 更新指标
        ious.update(batch_iou, num_images)
        f1_scores.update(batch_f1, num_images)
        dice_scores.update(batch_dice, num_images)

        csv_dict = {
        "Name": iter,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{batch_dice:.4f}",
        "Mean IoU": f"{batch_iou:.4f}",
        "Mean F1": f"{batch_f1:.4f}",
        "iters": iters
        }
        # 每条都写入csv
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
        
        # 累加
        ious.update(batch_iou, num_images)
        f1_scores.update(batch_f1, num_images)
        dice_scores.update(batch_dice, num_images)

        fabric.print(
            f'test: [{iter}] - [{iter}/{len(test_dataloader)}]: -- Mean Dice: [{dice_scores.avg:.4f}] Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
        )

        torch.cuda.empty_cache()

    fabric.print(f'Test [{iters}]:  Mean Dice: [{dice_scores.avg:.4f}]  -- Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    # 写入 CSV
    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "Mean IoU": f"{ious.avg:.4f}",
        "Mean F1": f"{f1_scores.avg:.4f}",
        "iters": iters
    }
    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)

    return ious.avg, f1_scores.avg, dice_scores.avg

def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)

def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(
    accelerator="auto",
    devices=num_devices,
    strategy="auto",
    precision="16-mixed",  # 启用 AMP 混合精度
)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)

        # 写入表头
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_head=cfg.csv_keys)
    
    sam_checkpoint = os.path.abspath("checkpoints/sam_vit_b_01ec64.pth")
    model_type = cfg.model.type
    print(model_type)
    print(sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(fabric.device)
    
    predictor = SamPredictor(sam)
    
    ours_predictor = configure_model_custom(predictor)
    params, _ = collect_params_custom(ours_predictor)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    
    # 加载数据集的函数
    load_datasets = call_load_dataset(cfg)

    Adapt(fabric, optimizer, cfg, predictor, load_datasets, name=cfg.name, iters=0)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    main(cfg)
    # multi_main(cfg)
    torch.cuda.empty_cache()
