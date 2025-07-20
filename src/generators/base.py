from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from ..data_loaders.data_structures import DataSample


class BaseGenerator(ABC):
    """数据生成器基类"""

    def __init__(self, config: Dict[str, Any], model_client=None):
        self.config = config
        self.model_client = model_client

    @abstractmethod
    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """生成单个数据样本

        Args:
            sample: 输入的数据样本

        Returns:
            生成的数据字典，如果生成失败返回None
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """返回输出数据的schema描述

        Returns:
            描述输出格式的字典
        """
        pass

    def validate_input(self, sample: DataSample) -> bool:
        """验证输入数据是否符合要求

        Args:
            sample: 输入数据样本

        Returns:
            是否通过验证
        """
        return True

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """验证输出数据是否符合要求

        Args:
            output: 生成的输出数据

        Returns:
            是否通过验证
        """
        return output is not None and isinstance(output, dict)


class PromptTemplate:
    """提示词模板基类"""

    def __init__(self, template: str):
        self.template = template

    def build_prompt(self, sample: DataSample) -> str:
        """构建提示词

        Args:
            sample: 输入数据样本

        Returns:
            构建好的提示词
        """
        # 基础实现：简单的字符串格式化
        # 子类可以重写这个方法实现复杂逻辑
        return self.template.format(**sample.metadata)


# 生成器注册机制
_GENERATOR_REGISTRY: Dict[str, Type[BaseGenerator]] = {}


def register_generator(name: str):
    """注册生成器装饰器

    Args:
        name: 生成器名称
    """

    def decorator(generator_class: Type[BaseGenerator]):
        if not issubclass(generator_class, BaseGenerator):
            raise ValueError(f"Generator {generator_class} must inherit from BaseGenerator")

        _GENERATOR_REGISTRY[name] = generator_class
        return generator_class

    return decorator


def get_generator(name: str) -> Optional[Type[BaseGenerator]]:
    """获取注册的生成器类

    Args:
        name: 生成器名称

    Returns:
        生成器类，如果不存在返回None
    """
    return _GENERATOR_REGISTRY.get(name)


def list_generators() -> Dict[str, Type[BaseGenerator]]:
    """列出所有注册的生成器

    Returns:
        生成器名称到类的映射
    """
    return _GENERATOR_REGISTRY.copy()


def create_generator(name: str, config: Dict[str, Any], model_client=None) -> Optional[BaseGenerator]:
    """创建生成器实例

    Args:
        name: 生成器名称
        config: 配置参数
        model_client: 模型客户端

    Returns:
        生成器实例，如果创建失败返回None
    """
    generator_class = get_generator(name)
    if generator_class is None:
        return None

    try:
        return generator_class(config, model_client)
    except Exception as e:
        print(f"Failed to create generator {name}: {e}")
        return None
