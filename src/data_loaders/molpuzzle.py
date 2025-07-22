import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
from .base import BaseDataLoader
from .registry import register_loader
from .data_structures import Dataset, DataSample


@register_loader("molpuzzle")
class MolPuzzleDataLoader(BaseDataLoader):
    """MolPuzzle 数据集加载器（每个样本为一个物质，聚合所有谱图）"""

    def can_handle(self, data_path: Path) -> bool:
        if data_path.suffix != '.json':
            return False
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                is_molecule = isinstance(first_item, dict) and "molecule_index" in first_item
                return is_molecule
            elif isinstance(data, dict) and "molecule_index" in data:
                return True
            else:
                return False
        except Exception:
            return False

    def load(self, data_path: Path) -> Dataset:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        if isinstance(data, list):
            for i, molecule_data in enumerate(data):
                sample = self._parse_molecule_all_spectra(molecule_data, f"{data_path.stem}_{i}")
                if sample:
                    samples.append(sample)
        elif isinstance(data, dict):
            sample = self._parse_molecule_all_spectra(data, data_path.stem)
            if sample:
                samples.append(sample)
        print(f"成功解析 {len(samples)} 个物质样本")
        return Dataset(samples=samples)

    def _parse_molecule_all_spectra(self, molecule_data: Dict[str, Any], molecule_id: str) -> Optional[DataSample]:
        try:
            molecule_info = {
                "molecule_index": molecule_data.get("molecule_index"),
                "smiles": molecule_data.get("smiles"),
                "formula": molecule_data.get("formula"),
            }
            images = {}
            for spectrum in molecule_data.get("spectra", []):
                spectrum_type = spectrum.get("spectrum_type", "Unknown")
                for path in spectrum.get("path", []):
                    # print(f"DEBUG: spectrum_type={spectrum_type}, raw_path={path}")
                    full_path = str(Path("seed_datasets/molpuzzle") / path)
                    # print(f"DEBUG: full_path={full_path}")
                    if os.path.isfile(full_path):
                        images[spectrum_type] = full_path
                    else:
                        print(f"警告: 图像文件不存在: [{spectrum_type}] {full_path}")
            if not images:
                return None
            text = f"Molecule: {molecule_info['formula']} ({molecule_info['smiles']})"
            return DataSample(
                id=molecule_id,
                text=text,
                images=images,
                metadata={
                    "molecule_info": molecule_info,
                    "molecule_id": molecule_id,
                },
            )
        except Exception as e:
            print(f"解析分子谱图失败: {e}")
            return None
