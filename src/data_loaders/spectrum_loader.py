import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
import re
from tqdm import tqdm

# Set custom HuggingFace cache directory BEFORE importing datasets
CUSTOM_HF_CACHE = "/mnt/shared-storage-user/xiejiaqing/huggingface"
if os.path.exists(CUSTOM_HF_CACHE):
    os.environ["HF_HOME"] = "/mnt/shared-storage-user/xiejiaqing/huggingface"
    os.environ["HF_DATASETS_CACHE"] = "/mnt/shared-storage-user/xiejiaqing/huggingface/datasets"
    print(f"Setting HuggingFace cache to: {os.environ['HF_DATASETS_CACHE']}")

from datasets import load_dataset
from PIL import Image as PILImage

from .base import BaseDataLoader
from .registry import register_loader
from .data_structures import Dataset, DataSample


@register_loader("spectrum")  # Register as generic spectrum loader
class SpectrumDataLoader(BaseDataLoader):
    """
    Universal data loader for spectroscopic datasets from Hugging Face.

    Supports all SpectrumWorld datasets with the following structure:
    - molecule_index: str
    - smiles: str
    - formula: str
    - {spectrum_type}_image: PIL.Image (e.g., ir_image, h-nmr_image, raman_image, etc.)

    This loader automatically detects available spectrum types from the data,
    making it dataset-agnostic and extensible to new spectrum types.
    """

    def can_handle(self, data_source: Union[str, str]) -> bool:
        """
        Check if this loader can handle the data source.
        Supports:
        - Hugging Face dataset IDs (format: 'username/dataset-name')
        - Local paths to HuggingFace cache directories containing .arrow files
        """
        if not isinstance(data_source, str):
            return False
        
        # Check if it's a local path to a directory containing .arrow files
        if os.path.isdir(data_source):
            arrow_files = list(Path(data_source).glob("*.arrow"))
            if arrow_files:
                return self._is_spectrum_hf_dataset_from_path(data_source)
        
        # Check if it's a HuggingFace dataset ID (contains '/')
        if '/' in data_source:
            return self._is_spectrum_hf_dataset(data_source)
        
        return False

    def _is_spectrum_hf_dataset(self, dataset_id: str) -> bool:
        """Check if the Hugging Face dataset is a spectroscopic dataset"""
        try:
            # Use the same cache directory as the main loading function
            cache_dir = os.path.join(CUSTOM_HF_CACHE, "datasets")
            
            # Try to load a small sample to check structure
            dataset = load_dataset(dataset_id, split="train[:1]", cache_dir=cache_dir)
            if len(dataset) == 0:
                return False

            sample = dataset[0]
            return self._check_sample_structure(sample)
        except Exception as e:
            print(f"Error checking HF dataset {dataset_id}: {e}")
            return False
    
    def _is_spectrum_hf_dataset_from_path(self, data_path: str) -> bool:
        """Check if a local path contains a spectroscopic dataset"""
        try:
            # Try to load from local path using arrow format
            arrow_files = list(Path(data_path).glob("*.arrow"))
            if not arrow_files:
                return False
            
            # Load first arrow file to check structure
            dataset = load_dataset("arrow", data_files=str(arrow_files[0]), split="train[:1]")
            if len(dataset) == 0:
                return False
            
            sample = dataset[0]
            return self._check_sample_structure(sample)
        except Exception as e:
            print(f"Error checking local dataset at {data_path}: {e}")
            return False
    
    def _check_sample_structure(self, sample: Dict[str, Any]) -> bool:
        """Check if a sample has the required structure for spectroscopic data"""
        # Check for required molecular fields (support multiple field name formats)
        # Support both old format (molecule_index, smiles) and new format (id, smiles_from_txt/smiles_from_txt_std)
        has_id = "id" in sample or "molecule_index" in sample
        has_smiles = "smiles" in sample or "smiles_from_txt" in sample or "smiles_from_txt_std" in sample
        has_formula = "formula" in sample
        has_required = has_id and has_smiles and has_formula

        # Check for spectrum image fields (exclude structure_image as it's not a spectrum)
        image_fields = [key for key in sample.keys() 
                      if key.endswith('_image') and key != 'structure_image']
        has_images = len(image_fields) > 0

        return has_required and has_images

    def load(self, data_source: Union[str, str], **kwargs) -> Dataset:
        """
        Load spectroscopic data from Hugging Face or local path.

        Supports:
        - Hugging Face dataset IDs (e.g., 'SpectrumWorld/dataset-name')
        - Local paths to directories containing .arrow files

        Args:
            data_source: Hugging Face dataset ID or local directory path
            **kwargs: Additional arguments:
                - max_samples: Maximum number of samples to load
                - split: Dataset split to load (default: "train")
        """
        if not isinstance(data_source, str):
            raise ValueError("data_source must be a string (dataset ID or local path)")

        max_samples = kwargs.get('max_samples', None)
        split = kwargs.get('split', 'train')
        # Optional starting offsets
        start_index = kwargs.get('start_index', 0)  # skip first N samples globally
        start_file_idx = kwargs.get('start_file_idx', 0)  # for local path: start from this arrow file index

        # Check if it's a local path
        if os.path.isdir(data_source):
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            return self._load_from_local_path(data_source, max_samples, split)
        else:
            # Assume it's a HuggingFace dataset ID
            return self._load_from_huggingface(data_source, max_samples, split)

    def _load_from_local_path(self, data_path: str, max_samples: Optional[int], split: str) -> Dataset:
        """Load spectroscopic data from local Arrow files"""
        print(f"Loading spectroscopic dataset from local path: {data_path}")
        
        try:
            # Find all .arrow files in the directory
            arrow_files = sorted(list(Path(data_path).glob("*.arrow")))
            if not arrow_files:
                raise ValueError(f"No .arrow files found in {data_path}")
            
            print(f"Found {len(arrow_files)} arrow file(s)")
            
            # Load dataset from arrow files
            if len(arrow_files) == 1:
                dataset = load_dataset("arrow", data_files=str(arrow_files[0]), split=split, streaming=True)
            else:
                # Multiple files - load all
                data_files = [str(f) for f in arrow_files]
                dataset = load_dataset("arrow", data_files=data_files, split=split, streaming=True)
            
            samples = []
            sample_count = 0
            
            for i, hf_sample in enumerate(dataset):
                if max_samples and sample_count >= max_samples:
                    break
                # Use id from data if available, otherwise use index
                sample_id_from_data = hf_sample.get("id") or hf_sample.get("molecule_index")
                if sample_id_from_data:
                    sample_id = f"local_{sample_id_from_data}"
                else:
                    sample_id = f"local_{i}"
                sample = self._parse_hf_sample(hf_sample, sample_id)
                if sample:
                    samples.append(sample)
                    sample_count += 1
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} samples, parsed {len(samples)} valid samples")
            
            print(f"Successfully parsed {len(samples)} samples from {i + 1} total samples processed")
            original_size = i + 1
            
            return Dataset(
                samples=samples,
                metadata={
                    "source": "local",
                    "data_path": data_path,
                    "split": split,
                    "original_size": original_size,
                },
            )
        
        except Exception as e:
            print(f"Error loading from local path: {e}")
            raise
=======
            return self._load_from_local_path(data_source, max_samples, split, start_index=start_index, start_file_idx=start_file_idx)
        else:
            # Assume it's a HuggingFace dataset ID
            return self._load_from_huggingface(data_source, max_samples, split, start_index=start_index)
>>>>>>> Stashed changes

=======
            return self._load_from_local_path(data_source, max_samples, split, start_index=start_index, start_file_idx=start_file_idx)
        else:
            # Assume it's a HuggingFace dataset ID
            return self._load_from_huggingface(data_source, max_samples, split, start_index=start_index)

>>>>>>> Stashed changes
    def _load_from_local_path(self, data_path: str, max_samples: Optional[int], split: str, start_index: int = 0, start_file_idx: int = 0) -> Dataset:
        """Load spectroscopic data from local Arrow files"""
        print(f"Loading spectroscopic dataset from local path: {data_path}")
        
        try:
            # Find all .arrow files in the directory
            arrow_files = sorted(list(Path(data_path).glob("*.arrow")))
            if not arrow_files:
                raise ValueError(f"No .arrow files found in {data_path}")
            
            print(f"Found {len(arrow_files)} arrow file(s)")
            if start_file_idx and 0 <= start_file_idx < len(arrow_files):
                arrow_files = arrow_files[start_file_idx:]
                print(f"Starting from arrow file index {start_file_idx}, remaining files: {len(arrow_files)}")
            
            # Load dataset from arrow files
            if len(arrow_files) == 1:
                dataset = load_dataset("arrow", data_files=str(arrow_files[0]), split=split, streaming=True)
            else:
                # Multiple files - load all
                data_files = [str(f) for f in arrow_files]
                dataset = load_dataset("arrow", data_files=data_files, split=split, streaming=True)
            
            samples = []
            sample_count = 0
            skipped = 0
            
            for i, hf_sample in enumerate(dataset):
                # Skip first start_index samples globally
                if start_index and skipped < start_index:
                    skipped += 1
                    if (skipped % 100) == 0:
                        print(f"Skipped {skipped} samples (start_index={start_index})")
                    continue
                if max_samples and sample_count >= max_samples:
                    break
                # Use id from data if available, otherwise use index
                sample_id_from_data = hf_sample.get("id") or hf_sample.get("molecule_index")
                if sample_id_from_data:
                    sample_id = f"local_{sample_id_from_data}"
                else:
                    sample_id = f"local_{i}"
                sample = self._parse_hf_sample(hf_sample, sample_id)
                if sample:
                    samples.append(sample)
                    sample_count += 1
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} samples, parsed {len(samples)} valid samples")
            
            print(f"Successfully parsed {len(samples)} samples from {i + 1} total samples processed")
            original_size = i + 1
            
            return Dataset(
                samples=samples,
                metadata={
                    "source": "local",
                    "data_path": data_path,
                    "split": split,
                    "original_size": original_size,
                },
            )
        
        except Exception as e:
            print(f"Error loading from local path: {e}")
            raise

    def _load_from_huggingface(self, dataset_id: str, max_samples: Optional[int], split: str, start_index: int = 0) -> Dataset:
        """Load spectroscopic data from Hugging Face dataset"""
        print(f"Loading spectroscopic dataset from Hugging Face: {dataset_id}")

        try:
            # Load dataset with custom cache directory
            # HuggingFace expects cache_dir to be the parent directory of all datasets
            # So we use the datasets directory directly, not the specific dataset folder
            cache_dir = os.path.join(CUSTOM_HF_CACHE, "datasets")
            print(f"Using cache directory: {cache_dir}")
            print(f"Cache directory exists: {os.path.exists(cache_dir)}")
            
            # Use streaming to avoid loading entire large datasets into memory
            # Note: streaming=True doesn't support split slicing, so we load the full split and limit later
            if os.path.exists(CUSTOM_HF_CACHE):
                dataset = load_dataset(dataset_id, split=split, cache_dir=cache_dir, streaming=True)
            else:
                dataset = load_dataset(dataset_id, split=split, streaming=True)

            samples = []
            sample_count = 0
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            
            for i, hf_sample in enumerate(dataset):
=======
            skipped = 0
            
            for i, hf_sample in enumerate(dataset):
=======
            skipped = 0
            
            for i, hf_sample in enumerate(dataset):
>>>>>>> Stashed changes
                # Skip first start_index samples globally
                if start_index and skipped < start_index:
                    skipped += 1
                    if (skipped % 100) == 0:
                        print(f"Skipped {skipped} samples (start_index={start_index})")
                    continue
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                if max_samples and sample_count >= max_samples:
                    break
                # Use id from data if available, otherwise use index
                sample_id_from_data = hf_sample.get("id") or hf_sample.get("molecule_index")
                if sample_id_from_data:
                    sample_id = f"{dataset_id}_{sample_id_from_data}"
                else:
                    sample_id = f"{dataset_id}_{i}"
                sample = self._parse_hf_sample(hf_sample, sample_id)
                if sample:
                    samples.append(sample)
                    sample_count += 1
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} samples, parsed {len(samples)} valid samples")
            
            print(f"Successfully parsed {len(samples)} samples from {i + 1} total samples processed")
            original_size = i + 1
            
            return Dataset(
                samples=samples,
                metadata={
                    "source": "huggingface",
                    "dataset_id": dataset_id,
                    "split": split,
                    "original_size": original_size,
                },
            )

        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            raise

    def _normalize_spectrum_type(self, field_name: str) -> str:
        """
        Normalize spectrum type from field name to standardized format.

        Examples:
            'ir_image' -> 'IR'
            'h-nmr_image' -> 'H-NMR'
            'c-nmr_image' -> 'C-NMR'
            'h_nmr_spectrum_image' -> 'H-NMR'
            'c_nmr_spectrum_image' -> 'C-NMR'
            'mass_image' -> 'MASS'
            'raman_image' -> 'RAMAN'
            'uv_image' -> 'UV'
            'hsqc_image' -> 'HSQC'
        """
        # Remove '_spectrum_image' or '_image' suffix
        if field_name.endswith('_spectrum_image'):
            spectrum_name = field_name.replace('_spectrum_image', '')
        elif field_name.endswith('_image'):
            spectrum_name = field_name.replace('_image', '')
        else:
            spectrum_name = field_name

        # Special handling for NMR types
        spectrum_name_lower = spectrum_name.lower()
        if 'h_nmr' in spectrum_name_lower or spectrum_name_lower.startswith('h-nmr'):
            return 'H-NMR'
        elif 'c_nmr' in spectrum_name_lower or spectrum_name_lower.startswith('c-nmr'):
            return 'C-NMR'
        
        # For other types, normalize: replace underscores/hyphens and convert to uppercase
        spectrum_type = spectrum_name.replace('_', '-').upper()
        return spectrum_type

    def _parse_hf_sample(self, hf_sample: Dict[str, Any], sample_id: str) -> Optional[DataSample]:
        """
        Parse a sample from Hugging Face dataset.

        Automatically detects and extracts all available spectrum images,
        making this method dataset-agnostic.
        
        Supports multiple field name formats:
        - Old format: molecule_index, smiles, formula
        - New format: id, smiles_from_txt/smiles_from_txt_std, formula
        """
        try:
            # Extract molecule information (support multiple field name formats)
            molecule_index = hf_sample.get("molecule_index") or hf_sample.get("id")
            
            # Try different SMILES field names (prefer standardized version)
            smiles = (hf_sample.get("smiles_from_txt_std") or 
                     hf_sample.get("smiles_from_txt") or 
                     hf_sample.get("smiles"))
            
            formula = hf_sample.get("formula")
            
            molecule_info = {
                "molecule_index": molecule_index,
                "smiles": smiles,
                "formula": formula,
            }
            
            # Also include additional fields if available
            if "iupac_name" in hf_sample:
                molecule_info["iupac_name"] = hf_sample.get("iupac_name")
            if "h_nmr_frequency" in hf_sample:
                molecule_info["h_nmr_frequency"] = hf_sample.get("h_nmr_frequency")
            if "h_nmr_solvent" in hf_sample:
                molecule_info["h_nmr_solvent"] = hf_sample.get("h_nmr_solvent")
            if "h_nmr_chemical_shift" in hf_sample:
                molecule_info["h_nmr_chemical_shift"] = hf_sample.get("h_nmr_chemical_shift")
            if "c_nmr_frequency" in hf_sample:
                molecule_info["c_nmr_frequency"] = hf_sample.get("c_nmr_frequency")
            if "c_nmr_solvent" in hf_sample:
                molecule_info["c_nmr_solvent"] = hf_sample.get("c_nmr_solvent")
            if "c_nmr_chemical_shift" in hf_sample:
                molecule_info["c_nmr_chemical_shift"] = hf_sample.get("c_nmr_chemical_shift")

            # Extract spectrum images (exclude structure_image as it's not a spectrum)
            images = {}
            image_fields = [key for key in hf_sample.keys() 
                          if key.endswith('_image') and key != 'structure_image']

            for img_field in image_fields:
                image_obj = hf_sample.get(img_field)
                if image_obj is None:
                    continue
                
                # Handle different image formats from Arrow files
                # HuggingFace datasets usually converts Image fields to PIL.Image automatically
                # But sometimes it might be bytes, so we handle both cases
                if isinstance(image_obj, PILImage.Image):
                    # Already a PIL Image, use directly
                    pil_image = image_obj
                elif isinstance(image_obj, bytes):
                    # Convert bytes to PIL Image
                    try:
                        from io import BytesIO
                        pil_image = PILImage.open(BytesIO(image_obj))
                    except Exception as e:
                        print(f"Warning: Failed to convert bytes to PIL Image for {img_field}: {e}")
                        continue
                else:
                    # Try to convert other formats (e.g., numpy array, etc.)
                    try:
                        pil_image = PILImage.fromarray(image_obj) if hasattr(image_obj, '__array__') else None
                        if pil_image is None:
                            print(f"Warning: Unsupported image format for {img_field}: {type(image_obj)}")
                            continue
                    except Exception as e:
                        print(f"Warning: Failed to convert image for {img_field}: {e}")
                        continue
                
                # Normalize spectrum type name
                spectrum_type = self._normalize_spectrum_type(img_field)
                images[spectrum_type] = pil_image

            if not images:
                print(f"Warning: No valid spectrum images found for sample {sample_id}")
                print(f"Available fields: {list(hf_sample.keys())}")
                print(f"Image fields found: {image_fields}")
                return None

            # Create descriptive text
            smiles_display = smiles if smiles else "Unknown"
            formula_display = formula if formula else "Unknown"
            text = f"Molecule: {formula_display} ({smiles_display})"

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
