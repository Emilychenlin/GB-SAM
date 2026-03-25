import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import re
from PIL import Image
import torchvision.transforms as transforms

class KvasirDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        self.image_filenames = sorted(
            [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(x.split('_')[-1])  
        )
        self.mask_filenames = sorted(
            [os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(x.split('_')[-1])  
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        img_path_jpg = os.path.join(self.image_dir, image_name + ".jpg")
        img_path_png = os.path.join(self.image_dir, image_name + ".png")

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image file {image_name} not found in either .jpg or .png format.")
        
        image = Image.open(img_path).convert("RGB")

        mask_path = os.path.join(self.mask_dir, mask_name + ".jpg")
        mask = Image.open(mask_path).convert("L")

        if np.sum(mask) == 0:
            raise ValueError(f"Mask for image {image_name} contains no valid (non-zero) regions.")

        if self.transform_image:
            image = self.transform_image(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)  # Convert numpy array to PIL Image
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, torch.tensor(mask, dtype=torch.uint8), image_name

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    img_ids = [item[2] for item in batch]
    return images, masks, img_ids


def load_dataset(data_path, 
                  mask_path,
                  batch_size=1):
    """

    para：
        - data_path
        - mask_path
        - batch_size
        - img_size:  224x224

    re：
        - train_loader: PyTorch DataLoader
    """
    
    # extract_data_dir = os.path.splitext(data_path)[0]  # "data/2017/ISIC-2017_Training_Data"
    # extract_mask_dir = os.path.splitext(mask_path)[0]  # "data/2017/ISIC-2017_Training_Part1_GroundTruth"

    # extract_zip(data_path, extract_data_dir)
    # extract_zip(mask_path, extract_mask_dir)

    extract_data_dir = data_path  # "data/2017/ISIC-2017_Training_Data"
    extract_mask_dir = mask_path  # "data/2017/ISIC-2017_Training_Part1_GroundTruth"

    transform_image = transforms.Compose([
        transforms.ToTensor(),                 
    ])
    transform_mask = transforms.Compose([
        transforms.ToTensor()
    ])

    
    dataset = KvasirDataset(image_dir=extract_data_dir, mask_dir=extract_mask_dir, transform_image=transform_image,transform_mask=transform_mask)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    print(f"load {len(dataset)} pictures")

    
    return train_loader


