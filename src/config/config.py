import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)


@dataclass
class ModelConfig:
    """模型配置类"""

    # DeepSeek API Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL")
    deepseek_model_name: str = os.getenv("DEEPSEEK_MODEL_NAME")

    # GPT-4o API Configuration
    gpt4o_api_key: str = os.getenv("GPT4O_API_KEY")
    gpt4o_base_url: str = os.getenv("GPT4O_BASE_URL")
    gpt4o_model_name: str = os.getenv("GPT4O_MODEL_NAME")

    # InternVL API Configuration
    internvl_api_key: str = os.getenv("INTERNVL_API_KEY")
    internvl_base_url: str = os.getenv("INTERNVL_BASE_URL", "https://internlm-chat.intern-ai.org.cn/puyu/api/v1")
    internvl_model_name: str = os.getenv("INTERNVL_MODEL_NAME", "internvl2_5-7b-chat")


@dataclass
class OmniGenConfig:
    """OmniGen 主配置类"""

    # 数据集配置
    dataset_path: str = "data"

    # 生成器配置
    generator_type: str = "benchmark"
    generator_config: Dict[str, Any] = field(default_factory=dict)

    # 模型配置
    model_type: str = "mock"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # 生成配置
    num_samples: int = 10
    output_path: str = "output"

    # 其他配置
    verbose: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """初始化后处理，自动填充模型配置"""
        if not self.model_config and self.model_type != "mock":
            self.model_config = self._get_model_config(self.model_type)

    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """根据模型类型获取默认配置"""
        model_configs = ModelConfig()

        if model_type.lower() == "internvl":
            return {
                "api_key": model_configs.internvl_api_key,
                "base_url": model_configs.internvl_base_url,
                "model_name": model_configs.internvl_model_name,
                "max_seq_len": 4096,
            }
        elif model_type.lower() == "deepseek":
            return {
                "api_key": model_configs.deepseek_api_key,
                "base_url": model_configs.deepseek_base_url,
                "model_name": model_configs.deepseek_model_name,
                "max_seq_len": 4096,
            }
        elif model_type.lower() == "gpt4o":
            return {
                "api_key": model_configs.gpt4o_api_key,
                "base_url": model_configs.gpt4o_base_url,
                "model_name": model_configs.gpt4o_model_name,
                "max_seq_len": 4096,
            }
        else:
            return {}

    @classmethod
    def from_file(cls, config_path: str) -> "OmniGenConfig":
        """从文件加载配置"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        return cls(**data)

    def to_file(self, config_path: str) -> None:
        """保存配置到文件"""
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
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []

        # 检查必要路径
        if not self.dataset_path:
            errors.append("dataset_path 不能为空")

        if not self.output_path:
            errors.append("output_path 不能为空")

        # 检查数值参数
        if self.num_samples <= 0:
            errors.append("num_samples 必须大于 0")

        if self.max_retries < 0:
            errors.append("max_retries 不能小于 0")

        # 检查模型配置
        if self.model_type != "mock" and self.model_config:
            api_key = self.model_config.get("api_key")
            if not api_key:
                errors.append(f"{self.model_type} 模型缺少 API Key")

        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False

        return True


# 保持向后兼容的别名
Config = ModelConfig
