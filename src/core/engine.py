import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ..config import OmniGenConfig
from ..data_loaders import Dataset, DataSample
from ..generators.base import create_generator
from ..models.base import BaseModel


class MockModel(BaseModel):
    """模拟模型，用于测试"""

    def __init__(self, path: str = "mock", max_seq_len: int = 2048):
        super().__init__(path, max_seq_len)

    def generate(self, prompt, max_out_len: int = 512) -> str:
        """生成模拟响应"""
        return '''
        {
            "question": "这是一个测试问题",
            "choices": ["选项A", "选项B", "选项C", "选项D", "选项E"],
            "answer": "选项A"
        }
        '''


class OmniGenEngine:
    """OmniGen 核心引擎"""

    def __init__(self, config: OmniGenConfig):
        self.config = config
        self.dataset: Optional[Dataset] = None
        self.generator = None
        self.model_client = None
        self.used_samples = set()  # 跟踪已使用的样本

        # 初始化组件
        self._init_model_client()
        self._init_generator()

    def _init_model_client(self):
        """初始化模型客户端"""
        model_type = self.config.model_type.lower()

        if model_type == "mock":
            self.model_client = MockModel()
        elif model_type == "internvl":
            try:
                from ..models import InternVL

                # 从配置中获取参数
                model_config = self.config.model_config
                self.model_client = InternVL(
                    model_name=model_config.get("model_name"),
                    api_key=model_config.get("api_key"),
                    base_url=model_config.get("base_url"),
                    max_seq_len=model_config.get("max_seq_len", 2048),
                )
            except ImportError as e:
                print(f"无法导入 InternVL: {e}")
                print("使用模拟客户端替代")
                self.model_client = MockModel()
            except Exception as e:
                print(f"InternVL 初始化失败: {e}")
                print("使用模拟客户端替代")
                self.model_client = MockModel()
        else:
            print(f"暂不支持模型类型: {model_type}，使用模拟客户端")
            self.model_client = MockModel()

    def _init_generator(self):
        """初始化生成器"""
        self.generator = create_generator(self.config.generator_type, self.config.generator_config, self.model_client)

        if self.generator is None:
            raise ValueError(f"无法创建生成器: {self.config.generator_type}")

    def load_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """加载数据集"""
        if dataset_path is None:
            dataset_path = self.config.dataset_path

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 使用数据加载器
        from ..data_loaders import create_loader_for_path

        loader = create_loader_for_path(dataset_path)

        if loader is None:
            raise ValueError(f"无法找到适合的加载器处理: {dataset_path}")

        self.dataset = loader.load(dataset_path)

        if self.config.verbose:
            stats = self.dataset.get_stats()
            print(f"数据集加载完成: {stats}")

        return self.dataset

    async def generate(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """生成数据"""
        if self.dataset is None:
            raise ValueError("请先加载数据集")

        if num_samples is None:
            num_samples = self.config.num_samples

        if self.config.verbose:
            print(f"开始生成 {num_samples} 个样本...")

        generated_data = []

        for i in range(num_samples):
            if self.config.verbose:
                print(f"正在生成第 {i+1}/{num_samples} 个样本...")

            # 随机选择一个输入样本
            sample = self._select_sample()

            retry_count = 0
            result = None

            while retry_count <= self.config.max_retries and result is None:
                try:
                    if self.config.verbose and retry_count > 0:
                        print(f"  重试第 {retry_count} 次...")

                    result = self.generator.generate_single(sample)

                    if result is not None:
                        generated_data.append(result)
                        if self.config.verbose:
                            print(f"  生成成功: {len(generated_data)}/{num_samples}")
                        break
                    else:
                        if self.config.verbose:
                            print(f"  生成返回 None，重试...")

                except Exception as e:
                    if self.config.verbose:
                        print(f"  生成异常 (重试 {retry_count}/{self.config.max_retries}): {e}")
                    retry_count += 1
                    continue

            if result is None:
                if self.config.verbose:
                    print(f"  样本 {i+1} 生成失败，已达到最大重试次数，跳过")
                continue

        if self.config.verbose:
            print(f"生成完成，共生成 {len(generated_data)} 个有效样本")

        return generated_data

    def _select_sample(self) -> DataSample:
        """选择一个输入样本，确保随机性和均衡性"""
        import random

        available_samples = [s for s in self.dataset.samples if s.id not in self.used_samples]

        # 如果所有样本都用过了，重置已使用集合
        if not available_samples:
            if self.config.verbose:
                print("所有样本都已使用过，重新开始选择")
            self.used_samples.clear()
            available_samples = self.dataset.samples

        # 随机选择一个样本
        selected_sample = random.choice(available_samples)
        self.used_samples.add(selected_sample.id)

        if self.config.verbose:
            print(f"选择样本: {selected_sample.id}")

        return selected_sample

    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None):
        """保存生成结果"""
        if output_path is None:
            output_path = self.config.output_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 根据文件扩展名选择保存格式
        if output_path.suffix.lower() == '.json':
            self._save_json(results, output_path)
        elif output_path.suffix.lower() in ['.csv']:
            self._save_csv(results, output_path)
        elif output_path.suffix.lower() == '.parquet':
            self._save_parquet(results, output_path)
        else:
            # 默认保存为JSON
            output_path = output_path.with_suffix('.json')
            self._save_json(results, output_path)

        if self.config.verbose:
            print(f"结果已保存到: {output_path}")

    def _save_json(self, results: List[Dict[str, Any]], output_path: Path):
        """保存为JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _save_csv(self, results: List[Dict[str, Any]], output_path: Path):
        """保存为CSV格式"""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')

    def _save_parquet(self, results: List[Dict[str, Any]], output_path: Path):
        """保存为Parquet格式"""
        df = pd.DataFrame(results)
        df.to_parquet(output_path, index=False)

    async def run(self) -> List[Dict[str, Any]]:
        """运行完整的生成流程"""
        # 加载数据集
        self.load_dataset()

        # 生成数据
        results = await self.generate()

        # 保存结果
        if results:
            self.save_results(results)

        return results
