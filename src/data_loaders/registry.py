from typing import Dict, Type, Optional
from pathlib import Path
from .base import BaseDataLoader

# global loader registry
_LOADER_REGISTRY: Dict[str, Type[BaseDataLoader]] = {}


def register_loader(name: str):
    def decorator(loader_class: Type[BaseDataLoader]):
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader {loader_class} must inherit from BaseDataLoader")

        _LOADER_REGISTRY[name] = loader_class
        return loader_class

    return decorator


def create_loader_for_path(data_path: Path) -> Optional[BaseDataLoader]:
    for loader_class in _LOADER_REGISTRY.values():
        try:
            loader = loader_class()
            if loader.can_handle(data_path):
                return loader
        except Exception as e:
            print(f"create loader failed: {e}")
            continue
    return None


def list_loaders() -> Dict[str, Type[BaseDataLoader]]:
    return _LOADER_REGISTRY.copy()
