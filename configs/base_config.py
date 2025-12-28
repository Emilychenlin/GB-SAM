base_config = {
    "eval_interval": 1,
    "ema_rate": 0.9999,
    "get_prompt": False,
    "split": True,
    "csv_keys": ["Name", "Prompt", "Mean Dice", "Mean IoU", "Mean F1", "iters"],
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": "vit_b",
        "checkpoint": "./checkpoints/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "Kvasir": {
            "train_dir": "data/Kvasir/train/images/",
            "val_dir": "data/Kvasir/val/images/",
            "test_dir": "data/Kvasir/test/images/",
            "train_list": "data/Kvasir/train/train.csv",
            "val_list": "data/Kvasir/val/val.csv",
            "test_list": "data/Kvasir/test/test.csv"
        },
        "BUSI": {
            "train_dir": "./data/BUSI/train/images/",
            "val_dir": "./data/BUSI/val/images/",
            "test_dir": "./data/BUSI/test/images/",
            "train_list": "./data/BUSI/train/train.csv",
            "val_list": "./data/BUSI/val/val.csv",
            "test_list": "./data/BUSI/test/test.csv"
        },
        "REFUGE_CUP": {
            "train_dir": "./data/REFUGE_CUP/train/images/",
            "val_dir": "./data/REFUGE_CUP/val/images/",
            "test_dir": "./data/REFUGE_CUP/test/images/",
            "train_list": "./data/REFUGE_CUP/train/train.csv",
            "val_list": "./data/REFUGE_CUP/val/val.csv",
            "test_list": "./data/REFUGE_CUP/test/test.csv"
        },
        "REFUGE_DISC": {
            "train_dir": "./data/REFUGE_DISC/train/images/",
            "val_dir": "./data/REFUGE_DISC/val/images/",
            "test_dir": "./data/REFUGE_DISC/test/images/",
            "train_list": "./data/REFUGE_DISC/train/train.csv",
            "val_list": "./data/REFUGE_DISC/val/val.csv",
            "test_list": "./data/REFUGE_DISC/test/test.csv"
        },
        "ISIC": {
            "train_dir": "./data/ISIC/train/",
            "val_dir": "./data/ISIC/val/",
            "test_dir": "./data/ISIC/test/",
            "train_list": "./data/ISIC/train/train.csv",
            "val_list": "./data/ISIC/val/val.csv",
            "test_list": "./data/ISIC/test/test.csv"
        },
    },
}
