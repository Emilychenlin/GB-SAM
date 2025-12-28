from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0",
    "batch_size": 1,
    "val_batchsize":1,
    "num_workers": 4,
    "num_epochs": 100,    
    "max_nums": 40,
    "num_points": 3,
    "eval_interval": 1,
    "dataset": "Kvasir",
    "prompt": "box",
    "out_dir": "output/Kvasir_sam", # 输出文件夹名称
    "name": "baseline",
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    },
    "model": {
        "type": "vit_b",
    },
}

cfg = Box(base_config)
cfg.merge_update(config)