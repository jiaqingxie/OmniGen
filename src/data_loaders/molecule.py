import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .base import BaseDataLoader
from .registry import register_loader
from .data_structures import Dataset, DataSample


@register_loader("molecule")
class MoleculeDataLoader(BaseDataLoader):
    """分子数据加载器"""

    def can_handle(self, data_path: Path) -> bool:
        """判断是否为分子数据文件"""
        if data_path.suffix != '.json':
            return False

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否为数组格式的分子数据
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                is_molecule = isinstance(first_item, dict) and "molecule_index" in first_item
                return is_molecule
            # 检查是否为单个分子数据
            elif isinstance(data, dict) and "molecule_index" in data:
                return True
            else:
                return False
        except Exception as e:
            return False

    def load(self, data_path: Path) -> Dataset:
        """加载分子数据，将每个谱图作为独立样本"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []

        if isinstance(data, list):
            # 处理数组格式
            for i, molecule_data in enumerate(data):
                molecule_samples = self._parse_molecule_spectra(molecule_data, f"{data_path.stem}_{i}")
                samples.extend(molecule_samples)
        elif isinstance(data, dict):
            # 处理单个分子数据
            molecule_samples = self._parse_molecule_spectra(data, data_path.stem)
            samples.extend(molecule_samples)

        print(f"成功解析 {len(samples)} 个谱图样本")
        return Dataset(samples=samples)

    def _parse_molecule_spectra(self, molecule_data: Dict[str, Any], molecule_id: str) -> List[DataSample]:
        """解析分子的所有谱图，每个谱图作为一个独立样本"""
        samples = []

        try:
            molecule_info = {
                "molecule_index": molecule_data.get("molecule_index"),
                "smiles": molecule_data.get("smiles"),
                "formula": molecule_data.get("formula"),
            }

            # 处理每个谱图
            for spectrum in molecule_data.get("spectra", []):
                spectrum_type = spectrum.get("spectrum_type", "Unknown")

                for path in spectrum.get("path", []):
                    # 构建完整路径
                    full_path = f"seed_datasets/molpuzzle/{path}"
                    if Path(full_path).exists():
                        # 为每个谱图创建独立样本
                        sample_id = f"{molecule_id}_{spectrum_type}"

                        # 构建文本描述
                        text = f"Molecule: {molecule_info['formula']} ({molecule_info['smiles']})"

                        sample = DataSample(
                            id=sample_id,
                            text=text,
                            images=[full_path],  # 只包含一个谱图
                            metadata={
                                "molecule_info": molecule_info,
                                "spectrum_type": spectrum_type,
                                "spectrum_path": path,
                                "molecule_id": molecule_id,
                            },
                        )
                        samples.append(sample)

        except Exception as e:
            print(f"解析分子谱图失败: {e}")

        return samples
