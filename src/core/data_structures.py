"""核心数据结构定义"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataSample:
    """统一数据样本格式"""

    id: str
    text: Optional[str] = None
    images: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_text(self) -> bool:
        """是否包含文本数据"""
        return self.text is not None and len(self.text.strip()) > 0

    def has_images(self) -> bool:
        """是否包含图像数据"""
        return self.images is not None and len(self.images) > 0

    def is_multimodal(self) -> bool:
        """是否为多模态数据"""
        return self.has_text() and self.has_images()

    def validate_images(self) -> List[str]:
        """验证图像路径并返回有效的路径"""
        if not self.has_images():
            return []

        valid_images = []
        for img_path in self.images:
            if Path(img_path).exists():
                valid_images.append(img_path)

        return valid_images


@dataclass
class Dataset:
    """统一数据集格式"""

    samples: List[DataSample]
    schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DataSample:
        return self.samples[index]

    def filter_by_modality(
        self, text_only: bool = False, image_only: bool = False, multimodal: bool = False
    ) -> "Dataset":
        """根据模态类型过滤数据"""
        filtered_samples = []

        for sample in self.samples:
            if text_only and not sample.has_images() and sample.has_text():
                filtered_samples.append(sample)
            elif image_only and sample.has_images() and not sample.has_text():
                filtered_samples.append(sample)
            elif multimodal and sample.is_multimodal():
                filtered_samples.append(sample)
            elif not (text_only or image_only or multimodal):
                filtered_samples.append(sample)

        return Dataset(samples=filtered_samples, schema=self.schema.copy(), metadata=self.metadata.copy())

    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        text_only = len([s for s in self.samples if not s.has_images() and s.has_text()])
        image_only = len([s for s in self.samples if s.has_images() and not s.has_text()])
        multimodal = len([s for s in self.samples if s.is_multimodal()])

        return {
            "total_samples": len(self.samples),
            "text_only": text_only,
            "image_only": image_only,
            "multimodal": multimodal,
        }
