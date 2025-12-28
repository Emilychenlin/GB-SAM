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
    """ Kvasir 皮肤病变数据集 """

    def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        # 支持 .jpg 和 .png 文件，并按文件名中的数字部分排序
        self.image_filenames = sorted(
            [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(x.split('_')[-1])  # 按文件名中的数字部分排序
        )
        self.mask_filenames = sorted(
            [os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))],
            key=lambda x: int(x.split('_')[-1])  # 按文件名中的数字部分排序
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        # 检查图片格式
        img_path_jpg = os.path.join(self.image_dir, image_name + ".jpg")
        img_path_png = os.path.join(self.image_dir, image_name + ".png")

        # 判断图片是否存在，并选择加载对应格式
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            raise FileNotFoundError(f"Image file {image_name} not found in either .jpg or .png format.")
        
        image = Image.open(img_path).convert("RGB")

        # 读取掩码（黑白图像）
        mask_path = os.path.join(self.mask_dir, mask_name + ".jpg")
        mask = Image.open(mask_path).convert("L")
        # # 指定保存的文件路径
        # save_path = 'mask_image.png'  # 请替换为您想要保存的路径

        # # 保存图像
        # mask.save(save_path)
        # exit()
        # 检查掩码是否有效
        if np.sum(mask) == 0:
            raise ValueError(f"Mask for image {image_name} contains no valid (non-zero) regions.")

        # 数据增强和预处理
        if self.transform_image:
            image = self.transform_image(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)  # Convert numpy array to PIL Image
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, torch.tensor(mask, dtype=torch.uint8), image_name

def custom_collate(batch):
    """处理包含字符串ID的批次数据"""
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    img_ids = [item[2] for item in batch]  # 保留字符串列表
    return images, masks, img_ids


def load_dataset(data_path, 
                  mask_path,
                  batch_size=1):
    """
    加载 Kvasir 数据集，并返回训练数据加载器。

    参数：
        - data_path: 训练数据集 ZIP 文件路径
        - mask_path: 掩码（分割标签）ZIP 文件路径
        - batch_size: 训练批量大小
        - img_size: 训练时的图像大小（默认 224x224）

    返回：
        - train_loader: PyTorch DataLoader
    """
    
    # # 定义解压路径
    # extract_data_dir = os.path.splitext(data_path)[0]  # "data/2017/ISIC-2017_Training_Data"
    # extract_mask_dir = os.path.splitext(mask_path)[0]  # "data/2017/ISIC-2017_Training_Part1_GroundTruth"

    # # 解压数据集
    # extract_zip(data_path, extract_data_dir)
    # extract_zip(mask_path, extract_mask_dir)

    # # 在解压后，查看文件数量
    # print(f"解压后的图像文件数量: {len(os.listdir(extract_mask_dir))}")
    extract_data_dir = data_path  # "data/2017/ISIC-2017_Training_Data"
    extract_mask_dir = mask_path  # "data/2017/ISIC-2017_Training_Part1_GroundTruth"

    # 定义数据预处理
    # transform_image = transforms.Compose([
    #     transforms.Resize((img_size_h, img_size_w)),  # 调整大小
    #     transforms.ToTensor(),                    # 转换为张量
    #     transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    # ])
    transform_image = transforms.Compose([
        transforms.ToTensor(),                    # 转换为张量
    ])
    transform_mask = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset = KvasirDataset(image_dir=extract_data_dir, mask_dir=extract_mask_dir, transform_image=transform_image,transform_mask=transform_mask)

    # # 创建数据加载器
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)

    print(f"加载 {len(dataset)} 张图片")

    
    return train_loader

# class KvasirDataset_supervised(Dataset):
#     """ Kvasir 皮肤病变数据集 """

#     def __init__(self, image_dir, transform_image=None):
#         self.image_dir = image_dir
#         self.transform_image = transform_image

#         # 支持 .jpg 和 .png 文件，并按文件名中的数字部分排序
#         self.image_filenames = sorted(
#             [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
#             key=lambda x: int(x.split('_')[-1])  # 按文件名中的数字部分排序
#         )

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         image_name = self.image_filenames[idx]

#         # 检查图片格式
#         img_path_jpg = os.path.join(self.image_dir, image_name + ".jpg")
#         img_path_png = os.path.join(self.image_dir, image_name + ".png")

#         # 判断图片是否存在，并选择加载对应格式
#         if os.path.exists(img_path_jpg):
#             img_path = img_path_jpg
#         elif os.path.exists(img_path_png):
#             img_path = img_path_png
#         else:
#             raise FileNotFoundError(f"Image file {image_name} not found in either .jpg or .png format.")
        
#         image = Image.open(img_path).convert("RGB")

#         # 数据增强和预处理
#         if self.transform_image:
#             image = self.transform_image(image)
            
#         return image, image_name


# def custom_collate_supervised(batch):
#     images = torch.stack([item[0] for item in batch])  # 堆叠图像张量
#     img_ids = [item[1] for item in batch]             # 收集文件名（字符串列表）
#     return images, img_ids


# def load_dataset(data_path, 
#                   batch_size=1):
#     """
#     加载 Kvasir 数据集，并返回训练数据加载器。

#     参数：
#         - data_path: 训练数据集 ZIP 文件路径
#         - mask_path: 掩码（分割标签）ZIP 文件路径
#         - batch_size: 训练批量大小
#         - img_size: 训练时的图像大小（默认 224x224）

#     返回：
#         - train_loader: PyTorch DataLoader
#     """
#     extract_data_dir = data_path  

#     transform_image = transforms.Compose([
#         transforms.ToTensor(),                    # 转换为张量
#     ])

#     # 创建数据集
#     dataset = KvasirDataset_supervised(image_dir=extract_data_dir,transform_image=transform_image)

#     # 创建数据加载器
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_supervised)

#     print(f"加载 {len(dataset)} 张图片")

#     return train_loader