from typing import Dict, Type, Optional, Union
from pathlib import Path
from .base import BaseDataLoader

# global loader registry
_LOADER_REGISTRY: Dict[str, Type[BaseDataLoader]] = {}


def register_loader(name: str):
    """
    Decorator to register a data loader.

    Args:
        name: Unique name for the loader
    """

    def decorator(loader_class: Type[BaseDataLoader]):
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader {loader_class} must inherit from BaseDataLoader")

        _LOADER_REGISTRY[name] = loader_class
        return loader_class

    return decorator


def create_loader_for_source(data_source: Union[str, Path]) -> Optional[BaseDataLoader]:
    """
    Create appropriate loader for the given data source.

    Args:
        data_source: Can be a file path, directory path, or Hugging Face dataset ID

    Returns:
        Appropriate loader instance or None if no loader can handle the source
    """
    for loader_name, loader_class in _LOADER_REGISTRY.items():
        try:
            loader = loader_class()
            if loader.can_handle(data_source):
                print(f"Using loader: {loader_name} for source: {data_source}")
                return loader
        except Exception as e:
            print(f"Failed to create loader {loader_name}: {e}")
            continue

    print(f"No loader found for source: {data_source}")
    return None


def create_loader_by_name(name: str) -> Optional[BaseDataLoader]:
    """
    Create loader by its registered name.

    Args:
        name: Registered loader name

    Returns:
        Loader instance or None if name not found
    """
    if name in _LOADER_REGISTRY:
        return _LOADER_REGISTRY[name]()
    return None


def list_loaders() -> Dict[str, Type[BaseDataLoader]]:
    """
    Get all registered loaders.

    Returns:
        Dictionary of loader name to loader class
    """
    return _LOADER_REGISTRY.copy()
