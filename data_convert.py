'''
data/Kvasir/
├── train/
│   ├── images/           
│   ├── masks/            
│   └── train.csv         
├── val/
│   ├── images/          
│   ├── masks/           
│   └── val.csv           
└── test/
    ├── images/          
    ├── masks/           
    └── test.csv         

'''
import os
import csv
import shutil

base_dir = 'data/REFUGE_DISC/val'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')
output_csv = os.path.join(base_dir, 'val.csv')

os.makedirs(image_dir, exist_ok=True)

for filename in os.listdir(mask_dir):
    name, ext = os.path.splitext(filename)
    if not name.endswith('_mask'):
        new_name = name + '_mask' + ext
    else:
        new_name = name + ext
    old_path = os.path.join(mask_dir, filename)
    new_path = os.path.join(image_dir, new_name)
    shutil.move(old_path, new_path)  # move

all_files = sorted(os.listdir(image_dir))

image_files = [f for f in all_files if '_mask' not in f]
mask_files = [f for f in all_files if '_mask' in f]

pairs = []
for mask_name in mask_files:
    mask_base_name = os.path.splitext(mask_name.replace('_mask', ''))[0]
    for image_name in image_files:
        image_base_name = os.path.splitext(image_name)[0]
        if image_base_name == mask_base_name:
            pairs.append((image_name, mask_name))
            break

# write CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'mask'])  
    for image, mask in pairs:
        writer.writerow([image, mask])

print(f"Success, {len(pairs)} in val.csv")
