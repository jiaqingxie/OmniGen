"""API 模型基类"""

from typing import Union, Dict, Any, Optional
from .base import BaseModel


class BaseAPIModel(BaseModel):
    """API 模型基类"""

    is_api: bool = True

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        # 调用父类构造函数，path 参数对 API 模型来说是 model_name
        super().__init__(path=model_name, max_seq_len=max_seq_len)

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        # 存储其他配置参数
        self.kwargs = kwargs

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        if not self.api_key or self.api_key.strip() == "":
            return False
        if not self.model_name or self.model_name.strip() == "":
            return False
        return True
