from abc import ABC, abstractmethod
from pathlib import Path
from .data_structures import Dataset


class BaseDataLoader(ABC):
    """Base data loader"""

    @abstractmethod
    def can_handle(self, data_path: Path) -> bool:
        """check if can handle the data path"""
        pass

    @abstractmethod
    def load(self, data_path: Path) -> Dataset:
        """load data and return Dataset"""
        pass
