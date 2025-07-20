import json
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample


@register_generator("benchmark")
class BenchmarkGenerator(BaseGenerator):
    """基准测试数据生成器"""

    def __init__(self, config: Dict[str, Any], model_client=None):
        super().__init__(config, model_client)

        # 从配置中获取提示词模板
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("配置中缺少 prompt_templates")

        # 问题类型配置
        self.question_types = config.get("question_types", ["type_cls"])
        self.current_type_index = 0

        # 其他配置参数
        self.num_choices = config.get("num_choices", 4)
        self.include_reasoning = config.get("include_reasoning", False)

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """生成单个基准测试样本"""
        if not self.validate_input(sample):
            return None

        try:
            # 获取当前问题类型
            question_type = self.question_types[self.current_type_index]

            # 检查是否有对应的模板
            if question_type not in self.prompt_templates:
                raise ValueError(f"配置中缺少问题类型 '{question_type}' 的模板")

            # 准备模型输入
            model_input = self._prepare_model_input(sample, question_type)

            # 调用模型生成
            if self.model_client is None:
                result = self._generate_mock_data(sample, question_type)
            else:
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

            # 更新问题类型索引，为下次生成做准备
            self.current_type_index = (self.current_type_index + 1) % len(self.question_types)

            return result

        except Exception as e:
            print(f"生成基准测试数据失败: {e}")
            return None

    def _prepare_model_input(self, sample: DataSample, question_type: str):
        """准备模型输入"""
        # 构建提示词
        prompt_text = self._build_prompt(sample, question_type)

        # 如果有图像，构建多模态输入
        if sample.has_images():
            return {"text": prompt_text, "images": sample.images}
        else:
            return prompt_text

    def _build_prompt(self, sample: DataSample, question_type: str) -> str:
        """构建提示词"""
        # 获取对应的模板
        template_str = self.prompt_templates[question_type]
        prompt_template = PromptTemplate(template_str)

        # 准备模板变量 - 只使用简单的元数据
        template_vars = {
            "num_choices": self.num_choices,
            "question_type": question_type,
        }

        # 安全地添加元数据中的简单字段
        if sample.metadata:
            for key, value in sample.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    template_vars[key] = value

        # 基础提示词
        try:
            base_prompt = prompt_template.template.format(**template_vars)
        except KeyError as e:
            print(f"模板格式化失败，缺少变量: {e}")
            # 使用原始模板
            base_prompt = template_str

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

    def _generate_mock_data(self, sample: DataSample, question_type: str) -> Dict[str, Any]:
        """生成模拟数据（用于测试）"""
        result = {
            "question": f"关于样本 {sample.id} 的{question_type}问题",
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
