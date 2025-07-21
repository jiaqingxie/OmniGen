import json
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample


@register_generator("benchmark")
class BenchmarkGenerator(BaseGenerator):
    """Benchmark data generator."""

    def __init__(self, config: Dict[str, Any], model_client=None):
        super().__init__(config, model_client)
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("Missing prompt_templates in config.")
        self.question_types = config.get("question_types", list(self.prompt_templates.keys()))
        self.current_type_index = 0
        self.include_reasoning = config.get("include_reasoning", False)

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single benchmark sample, print prompt, raw output, and parsed result."""
        if not self.validate_input(sample):
            return None
        try:
            question_type = self.question_types[self.current_type_index]
            template_info = self.prompt_templates[question_type]
            if isinstance(template_info, dict):
                template_str = template_info["template"]
                num_choices = template_info.get("num_choices", 4)
            else:
                template_str = template_info
                num_choices = 4
            model_input = self._prepare_model_input(sample, question_type, template_str)
            prompt_str = model_input["text"] if isinstance(model_input, dict) else model_input
            if self.model_client is None:
                raw_output = str(self._generate_mock_data(sample, question_type, num_choices))
                result = self._generate_mock_data(sample, question_type, num_choices)
            else:
                raw_output = self.model_client.generate(model_input, max_out_len=512)
                result = self._parse_response(raw_output, num_choices)
            # Print debug info
            debug_info = {
                "sample_id": sample.id,
                "question_type": question_type,
                "prompt": prompt_str,
                "raw_output": raw_output,
                "parsed_result": result,
            }
            print("\n[DEBUG] Benchmark Generation Info:")
            print(json.dumps(debug_info, ensure_ascii=False, indent=2))
            if result is None:
                return None
            if sample.has_images():
                result["image_path"] = sample.images
            if not self.validate_output(result, num_choices):
                return None
            self.current_type_index = (self.current_type_index + 1) % len(self.question_types)
            return result
        except Exception as e:
            print(f"Failed to generate benchmark data: {e}")
            return None

    def _prepare_model_input(self, sample: DataSample, question_type: str, template_str: str):
        prompt_text = self._build_prompt(sample, question_type, template_str)
        if sample.has_images():
            return {"text": prompt_text, "images": sample.images}
        else:
            return prompt_text

    def _build_prompt(self, sample: DataSample, question_type: str, template_str: str) -> str:
        prompt_template = PromptTemplate(template_str)
        template_vars = {
            "question_type": question_type,
        }
        if sample.metadata:
            for key, value in sample.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    template_vars[key] = value
        try:
            base_prompt = prompt_template.template.format(**template_vars)
        except KeyError as e:
            print(f"Prompt formatting failed, missing variable: {e}")
            base_prompt = template_str
        if sample.has_text():
            base_prompt += f"\n\nContent: {sample.text}"
        return base_prompt

    def _parse_response(self, response: str, num_choices: int) -> Optional[Dict[str, Any]]:
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx == -1 or end_idx == -1:
                return None
            json_str = response[start_idx : end_idx + 1]
            data = json.loads(json_str)
            required_fields = ["question", "choices", "answer"]
            if not all(field in data for field in required_fields):
                return None
            if not isinstance(data["choices"], list):
                return None
            if data["answer"] not in data["choices"]:
                return None
            return data
        except Exception:
            return None

    def _generate_mock_data(self, sample: DataSample, question_type: str, num_choices: int) -> Dict[str, Any]:
        result = {
            "question": f"Sample {sample.id} {question_type} question",
            "choices": [f"Option {chr(65+i)}" for i in range(num_choices)],
            "answer": "Option A",
        }
        if sample.has_images():
            result["image_path"] = sample.images
        return result

    def validate_input(self, sample: DataSample) -> bool:
        return sample.has_text() or sample.has_images()

    def validate_output(self, output: Dict[str, Any], num_choices: int) -> bool:
        if not super().validate_output(output):
            return False
        required_fields = ["question", "choices", "answer"]
        if not all(field in output for field in required_fields):
            return False
        if not isinstance(output["choices"], list):
            return False
        if output["answer"] not in output["choices"]:
            return False
        return True

    def get_output_schema(self) -> Dict[str, Any]:
        # 默认返回 type_cls 的 num_choices
        type_cls_info = self.prompt_templates.get("type_cls", {})
        num_choices = type_cls_info.get("num_choices", 5) if isinstance(type_cls_info, dict) else 5
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question text"},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": num_choices,
                    "maxItems": num_choices,
                    "description": "List of choices",
                },
                "answer": {"type": "string", "description": "Correct answer"},
                "image_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of image paths (optional)",
                },
            },
            "required": ["question", "choices", "answer"],
        }
