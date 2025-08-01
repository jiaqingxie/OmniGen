import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import re
from datasets import load_dataset
from PIL import Image as PILImage

from .base import BaseDataLoader
from .registry import register_loader
from .data_structures import Dataset, DataSample


@register_loader("molpuzzle")
class MolPuzzleDataLoader(BaseDataLoader):
    """
    Data loader for MolPuzzle dataset from Hugging Face.
    Supports loading MolPuzzle datasets with embedded images.
    """

    def can_handle(self, data_source: Union[str, str]) -> bool:
        """
        Check if this loader can handle the data source.
        Only handles Hugging Face dataset IDs with format 'username/dataset-name'.
        """
        if not isinstance(data_source, str) or '/' not in data_source:
            return False

        return self._is_molpuzzle_hf_dataset(data_source)

    def _is_molpuzzle_hf_dataset(self, dataset_id: str) -> bool:
        """Check if the Hugging Face dataset is a MolPuzzle dataset"""
        try:
            # Try to load a small sample to check structure
            dataset = load_dataset(dataset_id, split="train[:1]")
            if len(dataset) == 0:
                return False

            sample = dataset[0]
            # Check for MolPuzzle specific fields
            required_fields = ["molecule_index", "smiles", "formula"]
            has_required = all(field in sample for field in required_fields)

            # Check for image fields
            image_fields = [key for key in sample.keys() if key.endswith('_image')]
            has_images = len(image_fields) > 0

            return has_required and has_images
        except Exception as e:
            print(f"Error checking HF dataset {dataset_id}: {e}")
            return False

    def load(self, data_source: Union[str, str], **kwargs) -> Dataset:
        """
        Load MolPuzzle data from Hugging Face.

        Args:
            data_source: Hugging Face dataset ID (e.g., 'username/dataset-name')
            **kwargs: Additional arguments:
                - max_samples: Maximum number of samples to load
                - split: Dataset split to load (default: "train")
        """
        if not isinstance(data_source, str) or '/' not in data_source:
            raise ValueError("data_source must be a Hugging Face dataset ID (format: 'username/dataset-name')")

        max_samples = kwargs.get('max_samples', None)
        split = kwargs.get('split', 'train')

        return self._load_from_huggingface(data_source, max_samples, split)

    def _load_from_huggingface(self, dataset_id: str, max_samples: Optional[int], split: str) -> Dataset:
        """Load data from Hugging Face dataset"""
        print(f"Loading MolPuzzle dataset from Hugging Face: {dataset_id}")

        try:
            # Load dataset
            if max_samples:
                dataset = load_dataset(dataset_id, split=f"{split}[:{max_samples}]")
            else:
                dataset = load_dataset(dataset_id, split=split)

            print(f"Loaded {len(dataset)} samples from Hugging Face")

            samples = []
            for i, hf_sample in enumerate(dataset):
                sample = self._parse_hf_sample(hf_sample, f"{dataset_id}_{i}")
                if sample:
                    samples.append(sample)

            print(f"Successfully parsed {len(samples)} samples")
            return Dataset(
                samples=samples,
                metadata={
                    "source": "huggingface",
                    "dataset_id": dataset_id,
                    "split": split,
                    "original_size": len(dataset),
                },
            )

        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            raise

    def _parse_hf_sample(self, hf_sample: Dict[str, Any], sample_id: str) -> Optional[DataSample]:
        """Parse a sample from Hugging Face dataset"""
        try:
            # Extract molecule information
            molecule_info = {
                "molecule_index": hf_sample.get("molecule_index"),
                "smiles": hf_sample.get("smiles"),
                "formula": hf_sample.get("formula"),
            }

            # Extract images (they should be PIL Image objects from HF)
            images = {}
            image_fields = [key for key in hf_sample.keys() if key.endswith('_image')]

            for img_field in image_fields:
                image_obj = hf_sample.get(img_field)
                if image_obj is not None and isinstance(image_obj, PILImage.Image):
                    # Convert field name to spectrum type (e.g., 'ir_image' -> 'IR')
                    spectrum_type = img_field.replace('_image', '').upper()
                    # Handle special cases for spectrum type naming
                    if spectrum_type == 'C-NMR':
                        spectrum_type = 'C-NMR'
                    elif spectrum_type == 'H-NMR':
                        spectrum_type = 'H-NMR'

                    images[spectrum_type] = image_obj

            if not images:
                print(f"Warning: No valid images found for sample {sample_id}")
                return None

            # Create descriptive text
            text = f"Molecule: {molecule_info['formula']} ({molecule_info['smiles']})"

            return DataSample(
                id=sample_id,
                text=text,
                images=images,
                metadata={
                    "molecule_info": molecule_info,
                    "source": "huggingface",
                    "image_count": len(images),
                    "image_types": list(images.keys()),
                },
            )

        except Exception as e:
            print(f"Error parsing HF sample {sample_id}: {e}")
            return None
