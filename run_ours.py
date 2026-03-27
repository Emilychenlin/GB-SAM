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
    collect：
    1. image_encoder.blocks.11
    2. image_encoder.heatmap_proj
    3. image_encoder.anchor

    return：
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
    
    requires_grad=True：
    - image_encoder.blocks.11（the lasr Block）
    - image_encoder.heatmap_proj
    - image_encoder.anchor
    """
    # 
    predictor.model.eval()
    predictor.model.requires_grad_(False)

    # 
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

        # 
        pred_mask_dir = os.path.join(cfg.out_dir, "pred_masks")
        # matrices = os.path.join(cfg.out_dir, "matrices")
        os.makedirs(pred_mask_dir, exist_ok=True)

        # 
        images = images.to(fabric.device)

        # prompts
        prompts = get_prompts(cfg, bboxes, gt_masks)
        
        img = images[0].cpu()

        if img.dtype == torch.float32 or img.max() <= 1.0:
            img = img * 255.0

        # Step 2: to numpy 
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        # 2. to uint8 
        img = img.to(torch.uint8)
        # 3. to numpy，CHW -> HWC
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        # print("img.shape:", img_np.shape)
        
        # print("prompts.shape:", prompts.shape)
        # print("prompts[0].shape:", prompts[0].shape)
        box_coords = np.array(prompts[0])  
        # print("box_coords.shape:", box_coords.shape)
        box_coords = box_coords.reshape(1, 4) 
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
        # 
        optimizer.zero_grad()
        
        # save mask
        visualize_and_save_masks(pred_masks, pred_mask_dir, img_name)

        gt_masks = gt_masks[0]
        print("gt_masks.shape:", gt_masks.shape)
        if gt_masks.max() > 1:
            gt_bin = (gt_masks == 255).float()
        else:
            gt_bin = gt_masks.float()

        # to numpy
        gt_np = gt_bin[0].cpu().numpy().astype(np.uint8) * 255  
        # print("gt_np.shape:", gt_np.shape)
        # print("gt_np.max:", gt_np.max())
        # print("pred_masks.shape:", pred_masks.shape)
        # print("pred_masks.max:", pred_masks.max())
        # 从 (1, H, W) -> (H, W)
        pred_np = pred_masks[0]  # out batch 
        
        if isinstance(pred_np, torch.Tensor):
            pred_np = pred_np.cpu().numpy()
    
        pred_np = pred_np.astype(np.uint8) * 255 

        # Dice 
        batch_dice = calculate_dice(gt_np, pred_np)
        # mIoU 
        batch_iou = calculate_miou(gt_np, pred_np)

        #  Dice ~ F1-score
        batch_f1 = batch_dice

        # update
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
        # write csv
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}.csv"), csv_dict, csv_head=cfg.csv_keys)
        
        # +
        ious.update(batch_iou, num_images)
        f1_scores.update(batch_f1, num_images)
        dice_scores.update(batch_dice, num_images)

        fabric.print(
            f'test: [{iter}] - [{iter}/{len(test_dataloader)}]: -- Mean Dice: [{dice_scores.avg:.4f}] Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
        )

        torch.cuda.empty_cache()

    fabric.print(f'Test [{iters}]:  Mean Dice: [{dice_scores.avg:.4f}]  -- Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    # CSV
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
    precision="16-mixed",  
)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)

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
    
    load_datasets = call_load_dataset(cfg)

    Adapt(fabric, optimizer, cfg, predictor, load_datasets, name=cfg.name, iters=0)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    main(cfg)
    # multi_main(cfg)
    torch.cuda.empty_cache()
