'''
data/Kvasir/
├── train/
│   ├── images/           # 存放训练集原始图片（不包含 _mask）
│   ├── masks/            # 存放训练集对应的 mask 图片（_mask 后缀）
│   └── train.csv         # 训练集图片与 mask 的配对信息 CSV
├── val/
│   ├── images/           # 存放验证集原始图片
│   ├── masks/            # 存放验证集对应的 mask 图片
│   └── val.csv           # 验证集配对 CSV
└── test/
    ├── images/           # 存放测试集原始图片
    ├── masks/            # 存放测试集对应的 mask 图片
    └── test.csv          # 测试集配对 CSV

'''
import os
import csv
import shutil

# 路径配置
base_dir = 'data/REFUGE_DISC/val'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')
output_csv = os.path.join(base_dir, 'val.csv')

# 确保 images 目录存在
os.makedirs(image_dir, exist_ok=True)

# 1️⃣ 修改 mask 文件名并移动到 image_dir
for filename in os.listdir(mask_dir):
    name, ext = os.path.splitext(filename)
    if not name.endswith('_mask'):
        new_name = name + '_mask' + ext
    else:
        new_name = name + ext
    old_path = os.path.join(mask_dir, filename)
    new_path = os.path.join(image_dir, new_name)
    shutil.move(old_path, new_path)  # 移动文件

# 2️⃣ 获取所有 image 和 mask 文件
all_files = sorted(os.listdir(image_dir))

# 区分 image 和 mask（假设 image 不包含 _mask，mask 包含 _mask）
image_files = [f for f in all_files if '_mask' not in f]
mask_files = [f for f in all_files if '_mask' in f]

# 用名称匹配生成配对：根据去掉扩展名的名称进行匹配
pairs = []
for mask_name in mask_files:
    mask_base_name = os.path.splitext(mask_name.replace('_mask', ''))[0]
    for image_name in image_files:
        image_base_name = os.path.splitext(image_name)[0]
        if image_base_name == mask_base_name:
            pairs.append((image_name, mask_name))
            break

# 3️⃣ 写入 CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'mask'])  # 写入表头
    for image, mask in pairs:
        writer.writerow([image, mask])

print(f"✅ 处理完成，共写入 {len(pairs)} 对样本到 val.csv")