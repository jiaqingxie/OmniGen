import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


@dataclass
class ModelConfig:
    # InternVL API Configuration
    internvl_api_key: str = os.getenv("INTERNVL_API_KEY")
    internvl_base_url: str = os.getenv("INTERNVL_BASE_URL")
    internvl_model_name: str = os.getenv("INTERNVL_MODEL_NAME")


@dataclass
class OmniGenConfig:
    # dataset config
    dataset_path: str = "data"

    # generator config
    generator_type: str = "benchmark"
    generator_config: Dict[str, Any] = field(default_factory=dict)

    # model config
    model_type: str = "mock"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # other config
    num_samples: int = 10
    output_path: str = "output"
    verbose: bool = True
    max_retries: int = 3

    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        model_configs = ModelConfig()

        if model_type.lower() == "internvl":
            return {
                "api_key": model_configs.internvl_api_key,
                "base_url": model_configs.internvl_base_url,
                "model_name": model_configs.internvl_model_name,
                "max_seq_len": 4096,
            }
        else:
            return {}

    @classmethod
    def from_file(cls, config_path: str) -> "OmniGenConfig":
        """load config from file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"unsupported config file format: {config_path.suffix}")

        return cls(**data)

    def to_file(self, config_path: str) -> None:
        """save config to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_path": self.dataset_path,
            "generator_type": self.generator_type,
            "generator_config": self.generator_config,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "num_samples": self.num_samples,
            "output_path": self.output_path,
            "verbose": self.verbose,
            "max_retries": self.max_retries,
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, ensure_ascii=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"unsupported config file format: {config_path.suffix}")
