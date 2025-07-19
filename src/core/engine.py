"""核心生成引擎"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ..config import OmniGenConfig
from .data_structures import Dataset, DataSample
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
            "choices": ["选项A", "选项B", "选项C", "选项D"],
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
        """加载数据集

        Args:
            dataset_path: 数据集路径，如果为None则使用配置中的路径

        Returns:
            加载的数据集
        """
        if dataset_path is None:
            dataset_path = self.config.dataset_path

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 检查是否是单个文件还是目录
        if dataset_path.is_file():
            self.dataset = self._load_single_file(dataset_path)
        else:
            self.dataset = self._load_directory(dataset_path)

        if self.config.verbose:
            stats = self.dataset.get_stats()
            print(f"数据集加载完成: {stats}")

        return self.dataset

    def _load_single_file(self, file_path: Path) -> Dataset:
        """加载单个文件"""
        if file_path.suffix.lower() == '.json':
            return self._load_json_file(file_path)
        elif file_path.suffix.lower() in ['.csv']:
            return self._load_csv_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _load_json_file(self, file_path: Path) -> Dataset:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                sample = self._parse_data_item(item, f"{file_path.stem}_{i}")
                if sample:
                    samples.append(sample)
        elif isinstance(data, dict):
            sample = self._parse_data_item(data, file_path.stem)
            if sample:
                samples.append(sample)

        return Dataset(samples=samples)

    def _load_csv_file(self, file_path: Path) -> Dataset:
        """加载CSV文件"""
        df = pd.read_csv(file_path)
        samples = []

        for idx, row in df.iterrows():
            sample = DataSample(
                id=str(idx),
                text=row.get('text'),
                images=self._parse_image_paths(row.get('images')),
                metadata=row.to_dict(),
            )
            samples.append(sample)

        return Dataset(samples=samples)

    def _load_directory(self, dir_path: Path) -> Dataset:
        """加载目录中的数据"""
        # 简单实现：寻找所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        samples = []

        for img_path in dir_path.rglob('*'):
            if img_path.suffix.lower() in image_extensions:
                sample = DataSample(id=img_path.stem, images=[str(img_path)], metadata={"source_path": str(img_path)})
                samples.append(sample)

        return Dataset(samples=samples)

    def _parse_data_item(self, item: Dict[str, Any], default_id: str) -> Optional[DataSample]:
        """解析数据项"""
        try:
            sample_id = str(item.get('id', default_id))
            text = item.get('text')
            images = self._parse_image_paths(item.get('images'))
            metadata = {k: v for k, v in item.items() if k not in ['id', 'text', 'images']}

            return DataSample(id=sample_id, text=text, images=images, metadata=metadata)
        except Exception as e:
            print(f"解析数据项失败: {e}")
            return None

    def _parse_image_paths(self, images_data) -> Optional[List[str]]:
        """解析图像路径"""
        if images_data is None:
            return None

        if isinstance(images_data, str):
            return [images_data]
        elif isinstance(images_data, list):
            return [str(img) for img in images_data]
        else:
            return None

    async def generate(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """生成数据

        Args:
            num_samples: 生成样本数量，如果为None则使用配置中的数量

        Returns:
            生成的数据列表
        """
        if self.dataset is None:
            raise ValueError("请先加载数据集")

        if num_samples is None:
            num_samples = self.config.num_samples

        if self.config.verbose:
            print(f"开始生成 {num_samples} 个样本...")

        generated_data = []

        for i in range(num_samples):
            # 随机选择一个输入样本
            sample = self._select_sample()

            retry_count = 0
            result = None

            while retry_count <= self.config.max_retries and result is None:
                try:
                    # 生成器现在是同步的，但我们仍然在异步上下文中运行
                    result = self.generator.generate_single(sample)
                    if result is not None:
                        generated_data.append(result)
                        if self.config.verbose:
                            print(f"生成进度: {len(generated_data)}/{num_samples}")
                        break
                except Exception as e:
                    if self.config.verbose:
                        print(f"生成失败 (重试 {retry_count}/{self.config.max_retries}): {e}")
                    retry_count += 1

            if result is None and self.config.verbose:
                print(f"样本 {i+1} 生成失败，已达到最大重试次数")

        if self.config.verbose:
            print(f"生成完成，共生成 {len(generated_data)} 个有效样本")

        return generated_data

    def _select_sample(self) -> DataSample:
        """选择一个输入样本"""
        import random

        return random.choice(self.dataset.samples)

    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None):
        """保存生成结果

        Args:
            results: 生成的结果列表
            output_path: 输出路径，如果为None则使用配置中的路径
        """
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
        """运行完整的生成流程

        Returns:
            生成的结果列表
        """
        # 加载数据集
        self.load_dataset()

        # 生成数据
        results = await self.generate()

        # 保存结果
        if results:
            self.save_results(results)

        return results
