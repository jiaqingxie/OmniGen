"""数据加载器注册机制"""

from typing import Dict, Type, Optional
from pathlib import Path
from .base import BaseDataLoader

# 加载器注册表
_LOADER_REGISTRY: Dict[str, Type[BaseDataLoader]] = {}


def register_loader(name: str):
    """注册数据加载器装饰器"""

    def decorator(loader_class: Type[BaseDataLoader]):
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader {loader_class} must inherit from BaseDataLoader")

        _LOADER_REGISTRY[name] = loader_class
        return loader_class

    return decorator


def get_loader(name: str) -> Optional[Type[BaseDataLoader]]:
    """获取注册的加载器类"""
    return _LOADER_REGISTRY.get(name)


def list_loaders() -> Dict[str, Type[BaseDataLoader]]:
    """列出所有注册的加载器"""
    return _LOADER_REGISTRY.copy()


def create_loader_for_path(data_path: Path) -> Optional[BaseDataLoader]:
    """根据路径创建合适的加载器实例"""
    for loader_class in _LOADER_REGISTRY.values():
        try:
            loader = loader_class()
            if loader.can_handle(data_path):
                return loader
        except Exception as e:
            print(f"创建加载器失败: {e}")
            continue

    return None


def get_loader_for_path(data_path: Path) -> Optional[Type[BaseDataLoader]]:
    """根据路径获取合适的加载器类"""
    for loader_class in _LOADER_REGISTRY.values():
        try:
            loader = loader_class()
            if loader.can_handle(data_path):
                return loader_class
        except:
            continue

    return None
