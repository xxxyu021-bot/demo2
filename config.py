import json
import os
import torch


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg["device"] == "cuda":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 拼接绝对路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfg["config_name"] = os.path.splitext(
        os.path.basename(config_path)
    )[0]
    dataset_name = cfg["dataset_name"]
    data_cfg = cfg["data"][dataset_name]
    cfg["train_txt_path"] = os.path.join(
        root_dir,
        data_cfg["train_txt_path"]
    )
    cfg["dev_txt_path"] = os.path.join(
        root_dir,
        data_cfg["dev_txt_path"]
    )
    cfg["test_txt_path"] = os.path.join(
        root_dir,
        data_cfg["test_txt_path"]
    )

    return cfg
