import os
import base64
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from .base_api import BaseAPIModel
from openai import OpenAI


class InternVL(BaseAPIModel):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        # 设置默认值
        model_name = model_name or os.getenv("INTERNVL_MODEL_NAME")
        api_key = api_key or os.getenv("INTERNVL_API_KEY")
        base_url = base_url or os.getenv("INTERNVL_BASE_URL")

        # 初始化父类
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url, max_seq_len=max_seq_len, **kwargs)

        # 验证配置
        if not self.validate_config():
            raise ValueError("InternVL 配置无效。请设置 INTERNVL_API_KEY 环境变量或提供 api_key 参数。")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: Union[str, Dict[str, Any]], max_out_len: int = 512) -> str:
        """
        生成回复，支持文本和多模态输入

        Args:
            prompt: 输入提示，可以是：
                   - str: 纯文本提示
                   - Dict: 多模态提示，格式：
                     {
                         "text": "问题文本",
                         "images": [{"type": "image_url", "image_url": {"url": "data:..."}}]
                     }
            max_out_len: 最大输出长度

        Returns:
            生成的回复字符串
        """
        try:
            messages = self._prepare_messages(prompt)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_out_len,
                temperature=0.7,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"InternVL API 调用失败: {e}")

    def _prepare_messages(self, prompt: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备消息格式"""
        messages = []

        if isinstance(prompt, dict):
            # 多模态输入
            content = []

            # 添加文本内容
            text = prompt.get("text", "")
            if text:
                content.append({"type": "text", "text": text})

            # 添加图像内容
            images = prompt.get("images", [])
            for image_data in images:
                if isinstance(image_data, dict):
                    content.append(image_data)
                elif isinstance(image_data, str):
                    # 如果是图像路径，转换为 base64 格式
                    image_content = self._process_image_path(image_data)
                    if image_content:
                        content.append(image_content)

            messages.append({"role": "user", "content": content})

        else:
            # 纯文本输入
            text_content = str(prompt)
            messages.append({"role": "user", "content": text_content})

        return messages

    def _process_image_path(self, image_path: str) -> Optional[Dict[str, Any]]:
        """处理图像路径，转换为 API 需要的格式"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Warning: image file does not exist: {image_path}")
                return None

            # 读取图像并转换为 base64
            with open(path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # 获取图像格式
            image_format = path.suffix.lower().lstrip('.')
            if image_format == 'jpg':
                image_format = 'jpeg'

            return {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}

        except Exception as e:
            print(f"Failed to process image: {e}")
            return None

    def validate_config(self) -> bool:
        """验证配置"""
        return super().validate_config() and self.base_url is not None
