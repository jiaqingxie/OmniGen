from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
from .data_structures import Dataset


class BaseDataLoader(ABC):
    """Base class for all data loaders"""

    @abstractmethod
    def can_handle(self, data_source: Union[str, Path]) -> bool:
        """
        Check if this loader can handle the given data source.

        Args:
            data_source: Can be a file path, directory path, or Hugging Face dataset ID

        Returns:
            True if this loader can handle the data source
        """
        pass

    @abstractmethod
    def load(self, data_source: Union[str, Path], **kwargs) -> Dataset:
        """
        Load data from the given source.

        Args:
            data_source: Can be a file path, directory path, or Hugging Face dataset ID
            **kwargs: Additional arguments for loading (e.g., max_samples, split)

        Returns:
            Dataset object containing the loaded samples
        """
        pass
