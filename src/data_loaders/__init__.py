"""数据加载器模块"""

# 首先导入基础组件
from .base import BaseDataLoader
from .registry import register_loader, get_loader_for_path, create_loader_for_path, list_loaders
from .data_structures import DataSample, Dataset

# 然后导入具体的加载器（这会触发注册）
from .molecule import MoleculeDataLoader

# 调试：打印已注册的加载器
print(f"已注册的数据加载器: {list(list_loaders().keys())}")

__all__ = [
    'BaseDataLoader',
    'MoleculeDataLoader',
    'register_loader',
    'get_loader_for_path',
    'create_loader_for_path',
    'DataSample',
    'Dataset',
]
