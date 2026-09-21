import random
import numpy as np
import torch

#添加随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def print_config(cfg):
    print("==== Loaded Config ====")
    print(f"Device: {cfg['device']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Max seq len: {cfg['max_seq_len']}")
    print("=======================\n")
