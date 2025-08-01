import json
import os
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample
from PIL import Image as PILImage


@register_generator("benchmark")
class BenchmarkGenerator(BaseGenerator):
    """Benchmark data generator."""

    def __init__(self, config: Dict, model_client=None):
        super().__init__(config, model_client)
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("Missing prompt_templates in config.")
        self.question_types = config.get("question_types", list(self.prompt_templates.keys()))
        self.current_type_index = 0
        self.include_reasoning = config.get("include_reasoning", False)

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single benchmark sample."""
        if not self.validate_input(sample):
            print(f"❌ Sample {sample.id} failed validation")
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

            print(f"🔄 Generating {question_type} question for sample {sample.id}")
            model_input = self._prepare_model_input(sample, question_type, template_str)

            if self.model_client is None:
                print("📝 Using simple result generation (no model)")
                result = self._generate_simple_result(sample, question_type, num_choices)
            else:
                print("🤖 Using model for generation")
                raw_output = self.model_client.generate(model_input, max_out_len=512)
                result = self._parse_response(raw_output, num_choices, sample)

            if result is None:
                print(f"❌ Result is None for sample {sample.id}")
                return None

            print(f"✅ Successfully generated result for sample {sample.id}")
            self.current_type_index = (self.current_type_index + 1) % len(self.question_types)
            return result
        except Exception as e:
            print(f"❌ Exception in generate_single for sample {sample.id}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _prepare_model_input(self, sample: DataSample, question_type: str, template_str: str):
        prompt_text = self._build_prompt(sample, question_type, template_str)
        if sample.has_images():
            image_paths = list(sample.images.values())
            return {"text": prompt_text, "images": image_paths}
        else:
            return prompt_text

    def _build_prompt(self, sample: DataSample, question_type: str, template_str: str) -> str:
        spectra_list = "\n".join([f"{k}: available" for k in (sample.images or {}).keys()])
        prompt_template = PromptTemplate(template_str)
        template_vars = {
            "question_type": question_type,
            "spectra_list": spectra_list,
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

    def _parse_response(self, response: str, num_choices: int, sample: DataSample) -> Optional[Dict[str, Any]]:
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx == -1 or end_idx == -1:
                return None
            json_str = response[start_idx : end_idx + 1]
            data = json.loads(json_str)
            required_fields = ["selected_spectrum_type", "question", "choices", "answer"]
            if not all(field in data for field in required_fields):
                return None
            if not isinstance(data["choices"], list):
                return None
            if data["answer"] not in data["choices"]:
                return None

            # 保存选中的图像并添加路径
            spectrum_type = data["selected_spectrum_type"]
            image_path = self._save_and_get_image_path(sample, spectrum_type)
            if image_path:
                data["images"] = {spectrum_type: image_path}
            else:
                data["images"] = {}
            return data
        except Exception:
            return None

    def _generate_simple_result(self, sample: DataSample, question_type: str, num_choices: int) -> Dict[str, Any]:
        """生成简单的测试结果"""
        import random

        spectrum_types = list(sample.images.keys()) if sample.has_images() else ["IR"]
        selected_type = random.choice(spectrum_types)

        # 生成简单的问题和选项
        questions = {
            "type_cls": f"What type of spectrum is this?",
            "peak_identification": f"Which peak corresponds to the functional group in this spectrum?",
            "compound_identification": f"What compound does this spectrum represent?",
        }

        choices_map = {
            "type_cls": [
                "Infrared Spectrum (IR)",
                "Proton Nuclear Magnetic Resonance (H-NMR)",
                "Mass Spectrometry (MS)",
                "Carbon-13 Nuclear Magnetic Resonance (C-NMR)",
            ],
            "peak_identification": ["Peak A", "Peak B", "Peak C", "Peak D"],
            "compound_identification": ["Compound A", "Compound B", "Compound C", "Compound D"],
        }

        question = questions.get(question_type, f"Sample {sample.id} {question_type} question")
        choices = choices_map.get(question_type, [f"Option {chr(65+i)}" for i in range(num_choices)])

        # 根据选择的谱图类型确定答案
        answer = choices[0]  # 默认答案
        if question_type == "type_cls":
            type_mapping = {
                "IR": "Infrared Spectrum (IR)",
                "H-NMR": "Proton Nuclear Magnetic Resonance (H-NMR)",
                "MS": "Mass Spectrometry (MS)",
                "C-NMR": "Carbon-13 Nuclear Magnetic Resonance (C-NMR)",
            }
            answer = type_mapping.get(selected_type, choices[0])

        # 保存选中的图像并获取路径
        image_path = self._save_and_get_image_path(sample, selected_type)

        result = {
            "selected_spectrum_type": selected_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "images": {selected_type: image_path} if image_path else {},
        }
        return result

    def _save_and_get_image_path(self, sample: DataSample, spectrum_type: str) -> str:
        """保存指定类型的图像到临时文件夹并返回路径"""
        if not sample.has_images() or spectrum_type not in sample.images:
            return ""

        image_data = sample.images[spectrum_type]
        if not isinstance(image_data, PILImage.Image):
            return ""

        # 创建输出目录
        output_dir = "output/temp_images"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        filename = f"{sample.id}_{spectrum_type.lower().replace('-', '_')}.png"
        file_path = os.path.join(output_dir, filename)

        # 保存图像
        image_data.save(file_path, "PNG")
        print(f"Saved image: {file_path}")

        return file_path

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
        return {
            "type": "object",
            "properties": {
                "selected_spectrum_type": {
                    "type": "string",
                    "description": "Selected spectrum type used for the question.",
                },
                "question": {"type": "string", "description": "Question text"},
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 10,
                    "description": "List of choices",
                },
                "answer": {"type": "string", "description": "Correct answer"},
                "images": {
                    "type": "object",
                    "description": "Selected spectrum type and its image path",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["selected_spectrum_type", "question", "choices", "answer"],
        }
