import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from datasets.tools import ResizeAndPad, collate_fn, get_largest_mask


class KvasirDataset(Dataset):
    def __init__(self, cfg, root_dir, list_file, transform=None):
        self.cfg = cfg
        df = pd.read_csv(os.path.join(list_file), encoding='gbk')
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, name)
        image = cv2.imread(image_path)
        if image is None:
           raise FileNotFoundError(f"[KvasirDataset] failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.cfg.get_prompt:
            image_info = {}
            height, width, _ = image.shape
            image_info["file_path"] = image_path
            image_info["height"] = height
            image_info["width"] = width
            return idx, image_info, image

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
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

        if self.transform:
            file_names = os.path.splitext(os.path.basename(name))[0]
            # file_names = os.path.basename(name)
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float(), file_names

def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    test = KvasirDataset(
        cfg,
        root_dir=cfg.datasets.Kvasir.test_dir,
        list_file=cfg.datasets.Kvasir.test_list,
        transform=transform,
    )
    test_dataloader = DataLoader(
        test,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return test_dataloader
