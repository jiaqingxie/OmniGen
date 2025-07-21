import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import random
from ..config import OmniGenConfig
from ..data_loaders import Dataset, DataSample, create_loader_for_path
from ..generators.base import create_generator
from ..models.base import BaseModel


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, path: str = "mock", max_seq_len: int = 2048):
        super().__init__(path, max_seq_len)

    def generate(self, prompt, max_out_len: int = 512) -> str:
        return '''
        {
            "question": "This is a test question",
            "choices": ["A", "B", "C", "D", "E"],
            "answer": "A"
        }
        '''


class OmniGenEngine:
    """OmniGen core engine: manages model, generator, data loading, generation, and saving."""

    def __init__(self, config: OmniGenConfig):
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.generator = None
        self.model_client = None
        self.used_samples = set()
        self._init_model_client()
        self._init_generator()

    def _init_model_client(self):
        """Initialize model client."""
        model_type = self.config.model_type.lower()
        if model_type == "mock":
            self.model_client = MockModel()
        elif model_type == "internvl":
            try:
                from ..models import InternVL

                model_config = self.config.model_config
                self.model_client = InternVL(
                    model_name=model_config.get("model_name"),
                    api_key=model_config.get("api_key"),
                    base_url=model_config.get("base_url"),
                    max_seq_len=model_config.get("max_seq_len", 2048),
                )
            except ImportError as e:
                print(f"[WARN] Failed to import InternVL: {e}. Using MockModel instead.")
                self.model_client = MockModel()
            except Exception as e:
                print(f"[WARN] InternVL initialization failed: {e}. Using MockModel instead.")
                self.model_client = MockModel()
        else:
            print(f"[WARN] Unsupported model type: {model_type}. Using MockModel instead.")
            self.model_client = MockModel()

    def _init_generator(self):
        """Initialize generator."""
        self.generator = create_generator(self.config.generator_type, self.config.generator_config, self.model_client)
        if self.generator is None:
            raise ValueError(f"Cannot create generator: {self.config.generator_type}")

    def load_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """Load dataset from path."""
        if dataset_path is None:
            dataset_path = self.config.dataset_path
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        loader = create_loader_for_path(dataset_path)
        if loader is None:
            raise ValueError(f"No suitable loader found for: {dataset_path}")
        self.dataset = loader.load(dataset_path)
        if self.config.verbose:
            stats = self.dataset.get_stats()
            print(f"Dataset loaded: {stats}")
        return self.dataset

    async def generate(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate data samples."""
        if self.dataset is None:
            raise ValueError("Please load dataset first.")
        if num_samples is None:
            num_samples = self.config.num_samples
        if self.config.verbose:
            print(f"Generating {num_samples} samples...")
        generated_data = []
        for i in range(num_samples):
            if self.config.verbose:
                print(f"Generating sample {i+1}/{num_samples} ...")
            sample = self._select_sample()
            result = await self._generate_with_retry(sample)
            if result is not None:
                generated_data.append(result)
                if self.config.verbose:
                    print(f"  Success: {len(generated_data)}/{num_samples}")
            else:
                if self.config.verbose:
                    print(f"  Failed to generate sample {i+1}, skipped after max retries.")
        if self.config.verbose:
            print(f"Generation finished. {len(generated_data)} valid samples generated.")
        return generated_data

    async def _generate_with_retry(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single sample with retry logic."""
        retry_count = 0
        result = None
        while retry_count <= self.config.max_retries and result is None:
            try:
                if self.config.verbose and retry_count > 0:
                    print(f"  Retry {retry_count} ...")
                maybe_result = self.generator.generate_single(sample)
                if maybe_result is not None:
                    result = maybe_result
                    break
                else:
                    if self.config.verbose:
                        print("  Generator returned None, retrying ...")
            except Exception as e:
                if self.config.verbose:
                    print(f"  Exception during generation (retry {retry_count}/{self.config.max_retries}): {e}")
            retry_count += 1
        return result

    def _select_sample(self) -> DataSample:
        """Randomly select a sample, ensuring balanced usage."""
        available_samples = [s for s in self.dataset.samples if s.id not in self.used_samples]
        if not available_samples:
            if self.config.verbose:
                print("All samples used, resetting usage tracker.")
            self.used_samples.clear()
            available_samples = self.dataset.samples
        selected_sample = random.choice(available_samples)
        self.used_samples.add(selected_sample.id)
        if self.config.verbose:
            print(f"Selected sample: {selected_sample.id}")
        return selected_sample

    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None):
        """Save generated results to file."""
        if output_path is None:
            output_path = self.config.output_path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_map = {
            '.json': self._save_json,
            '.csv': self._save_csv,
            '.parquet': self._save_parquet,
        }
        ext = output_path.suffix.lower()
        save_func = save_map.get(ext, self._save_json)
        save_func(results, output_path)
        if self.config.verbose:
            print(f"Results saved to: {output_path}")

    def _save_json(self, results: List[Dict[str, Any]], output_path: Path):
        """Save as JSON format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _save_csv(self, results: List[Dict[str, Any]], output_path: Path):
        """Save as CSV format."""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

    def _save_parquet(self, results: List[Dict[str, Any]], output_path: Path):
        """Save as Parquet format."""
        df = pd.DataFrame(results)
        df.to_parquet(output_path, index=False)

    async def run(self) -> List[Dict[str, Any]]:
        """Run the full generation pipeline."""
        self.load_dataset()
        results = await self.generate()
        if results:
            self.save_results(results)
        return results
