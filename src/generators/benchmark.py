"""基准测试数据生成器"""

import json
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..core.data_structures import DataSample


@register_generator("benchmark")
class BenchmarkGenerator(BaseGenerator):
    """基准测试数据生成器

    生成标准的多选题格式，包含：
    - question: 问题文本
    - choices: 选项列表
    - answer: 正确答案
    - image_path: 图像路径（如果有）
    """

    def __init__(self, config: Dict[str, Any], model_client=None):
        super().__init__(config, model_client)

        # 从配置中获取提示词模板
        template_str = config.get("prompt_template", self._default_template())
        self.prompt_template = PromptTemplate(template_str)

        # 其他配置参数
        self.num_choices = config.get("num_choices", 4)
        self.include_reasoning = config.get("include_reasoning", False)

    def _default_template(self) -> str:
        """默认提示词模板"""
        return """
请根据提供的内容生成一个多选题。

要求：
1. 生成一个清晰的问题
2. 提供{num_choices}个选项
3. 确保只有一个正确答案
4. 选项应该具有合理的干扰性

请以JSON格式回复：
{{
    "question": "问题内容",
    "choices": ["选项A", "选项B", "选项C", "选项D"],
    "answer": "正确答案"
}}
"""

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """生成单个基准测试样本"""
        if not self.validate_input(sample):
            return None

        try:
            # 调用模型生成
            if self.model_client is None:
                # 如果没有模型客户端，返回模拟数据
                return self._generate_mock_data(sample)

            # 准备模型输入
            model_input = self._prepare_model_input(sample)

            # 调用模型（同步调用）
            response = self.model_client.generate(model_input, max_out_len=512)

            # 解析响应
            result = self._parse_response(response)

            if result is None:
                return None

            # 添加图像路径（如果有）
            if sample.has_images():
                result["image_path"] = sample.images

            # 验证输出
            if not self.validate_output(result):
                return None

            return result

        except Exception as e:
            print(f"生成基准测试数据失败: {e}")
            return None

    def _prepare_model_input(self, sample: DataSample):
        """准备模型输入"""
        # 构建提示词
        prompt_text = self._build_prompt(sample)

        # 如果有图像，构建多模态输入
        if sample.has_images():
            return {"text": prompt_text, "images": sample.images}  # 直接传递图像路径，模型会处理
        else:
            # 纯文本输入
            return prompt_text

    def _build_prompt(self, sample: DataSample) -> str:
        """构建提示词"""
        # 准备模板变量
        template_vars = {"num_choices": self.num_choices, **sample.metadata}

        # 基础提示词
        base_prompt = self.prompt_template.template.format(**template_vars)

        # 如果有文本内容，添加到提示词中
        if sample.has_text():
            base_prompt += f"\n\n内容：{sample.text}"

        return base_prompt

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析模型响应"""
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}')

            if start_idx == -1 or end_idx == -1:
                return None

            json_str = response[start_idx : end_idx + 1]
            data = json.loads(json_str)

            # 验证必要字段
            required_fields = ["question", "choices", "answer"]
            if not all(field in data for field in required_fields):
                return None

            # 验证选项格式
            if not isinstance(data["choices"], list) or len(data["choices"]) != self.num_choices:
                return None

            # 验证答案是否在选项中
            if data["answer"] not in data["choices"]:
                return None

            return data

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _generate_mock_data(self, sample: DataSample) -> Dict[str, Any]:
        """生成模拟数据（用于测试）"""
        result = {
            "question": f"关于样本 {sample.id} 的问题",
            "choices": ["选项A", "选项B", "选项C", "选项D"],
            "answer": "选项A",
        }

        if sample.has_images():
            result["image_path"] = sample.images

        return result

    def validate_input(self, sample: DataSample) -> bool:
        """验证输入数据"""
        # 至少需要有文本或图像
        return sample.has_text() or sample.has_images()

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """验证输出数据"""
        if not super().validate_output(output):
            return False

        # 检查必要字段
        required_fields = ["question", "choices", "answer"]
        if not all(field in output for field in required_fields):
            return False

        # 检查选项数量
        if len(output["choices"]) != self.num_choices:
            return False

        # 检查答案是否在选项中
        if output["answer"] not in output["choices"]:
            return False

        return True

    def get_output_schema(self) -> Dict[str, Any]:
        """返回输出数据的schema"""
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "问题文本"},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": self.num_choices,
                    "maxItems": self.num_choices,
                    "description": "选项列表",
                },
                "answer": {"type": "string", "description": "正确答案"},
                "image_path": {"type": "array", "items": {"type": "string"}, "description": "图像路径列表（可选）"},
            },
            "required": ["question", "choices", "answer"],
        }
