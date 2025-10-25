# pointcept/datasets/builder.py
import torch
from functools import partial
from torch.utils.data import DataLoader
from pointcept.utils.registry import Registry

DATASETS = Registry('datasets')

def build_dataset(cfg):
    """构建数据集：接收数据集配置（如cfg['data']['train']）"""
    return DATASETS.build(cfg)