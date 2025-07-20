"""数据加载器基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from .data_structures import Dataset


class BaseDataLoader(ABC):
    """数据加载器基类"""

    @abstractmethod
    def can_handle(self, data_path: Path) -> bool:
        """判断是否能处理该数据路径"""
        pass

    @abstractmethod
    def load(self, data_path: Path) -> Dataset:
        """加载数据并返回 Dataset"""
        pass
