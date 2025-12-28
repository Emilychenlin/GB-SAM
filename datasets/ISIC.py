import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask, get_largest_mask


class ISICDataset(Dataset):
    def __init__(self, cfg, root_dir, list_file, transform=None, if_self_training=False):
        self.cfg = cfg
        df = pd.read_csv(os.path.join(list_file), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transform = transform

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.img_dir, name)
        image = cv2.imread(image_path)
        if image is None:
           raise FileNotFoundError(f"[ISICDataset] failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = image_path
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.mask_dir, label_name)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        masks = []
        bboxes = []
        categories = []
        # gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        # areas = gt_masks.reshape(gt_masks.shape[0], -1).sum(axis=1)
        # max_idx = areas.argmax()

        # gt_mask = gt_masks[max_idx:max_idx+1]  # 取第max_idx个掩码，保持维度为(1, H, W)
        gt_mask_tensor = torch.tensor(gt_mask[None, :, :])
        largest_mask = get_largest_mask(gt_mask_tensor).squeeze(0).numpy().astype(np.uint8)

        masks.append(largest_mask)
        x, y, w, h = cv2.boundingRect(largest_mask)
        bboxes.append([x, y, x + w, y + h])
        categories.append("0")

        # print("gt_masks.sum():", gt_masks.sum())
        # print("(gt_mask > 0).sum():", (gt_mask > 0).sum())
        # assert gt_masks.sum() == (gt_mask > 0).sum()

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float()

        elif self.cfg.visual:
            file_name = os.path.splitext(os.path.basename(name))[0]
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return file_name, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                file_names = os.path.splitext(os.path.basename(name))[0]
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float(), file_names

def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    test = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.test_dir,
        list_file=cfg.datasets.ISIC.test_list,
        transform=transform,
    )
    val = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.val_dir,
        list_file=cfg.datasets.ISIC.val_list,
        transform=transform,
    )
    train = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.train_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        if_self_training=cfg.augment,
    )
    test_dataloader = DataLoader(
        test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader

def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = ISICDataset(
        cfg,
        root_dir=cfg.datasets.ISIC.train_dir,
        list_file=cfg.datasets.ISIC.train_list,
        transform=transform,
        if_self_training=cfg.augment,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return train_dataloader