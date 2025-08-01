from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from PIL import Image as PILImage


@dataclass
class DataSample:
    """Data sample for multimodal data"""

    id: str
    text: Optional[str] = None
    images: Optional[Dict[str, Union[str, PILImage.Image]]] = None  # 支持文件路径或PIL图像对象
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_text(self) -> bool:
        """check if has text"""
        return self.text is not None and len(self.text.strip()) > 0

    def has_images(self) -> bool:
        """check if has images"""
        return self.images is not None and len(self.images) > 0

    def is_multimodal(self) -> bool:
        """check if is multimodal"""
        return self.has_text() and self.has_images()

    def get_image_count(self) -> int:
        """get number of images"""
        return len(self.images) if self.images else 0

    def get_image_types(self) -> List[str]:
        """get list of image types/keys"""
        return list(self.images.keys()) if self.images else []


@dataclass
class Dataset:
    """Dataset containing multiple data samples"""

    samples: List[DataSample]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DataSample:
        return self.samples[index]

    def get_stats(self) -> Dict[str, Any]:
        """get dataset statistics"""
        text_only = len([s for s in self.samples if s.has_text() and not s.has_images()])
        image_only = len([s for s in self.samples if s.has_images() and not s.has_text()])
        multimodal = len([s for s in self.samples if s.is_multimodal()])

        total_images = sum(s.get_image_count() for s in self.samples)

        return {
            "total_samples": len(self.samples),
            "text_only": text_only,
            "image_only": image_only,
            "multimodal": multimodal,
            "total_images": total_images,
        }
