import json
import os
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample
from PIL import Image as PILImage


@register_generator("benchmark")
class BenchmarkGenerator(BaseGenerator):
    """Benchmark data generator for creating multiple-choice questions."""

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
            return None

        if self.model_client is None:
            raise ValueError("Model client is required for benchmark generation")

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

            try:
                raw_output = self.model_client.generate(model_input, max_out_len=512)
                if raw_output:
                    result = self._parse_response(raw_output, num_choices, sample)
                    if result:
                        self.current_type_index = (self.current_type_index + 1) % len(self.question_types)
                        return result
            except Exception as e:
                pass  # Continue to return None

            return None
        except Exception as e:
            return None

    def _prepare_model_input(self, sample: DataSample, question_type: str, template_str: str):
        """Prepare input for the model including text prompt and images."""
        prompt_text = self._build_prompt(sample, question_type, template_str)
        if sample.has_images():
            image_paths = list(sample.images.values())
            return {"text": prompt_text, "images": image_paths}
        else:
            return prompt_text

    def _build_prompt(self, sample: DataSample, question_type: str, template_str: str) -> str:
        """Build the text prompt using template and sample data."""
        if sample.has_images():
            available_data = "\n".join([f"{k}: available" for k in sample.images.keys()])
        else:
            available_data = "text data: available"

        prompt_template = PromptTemplate(template_str)
        template_vars = {
            "question_type": question_type,
            "available_data": available_data,
        }
        if sample.metadata:
            for key, value in sample.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    template_vars[key] = value
        try:
            base_prompt = prompt_template.template.format(**template_vars)
        except KeyError as e:
            base_prompt = template_str
        if sample.has_text():
            base_prompt += f"\n\nContent: {sample.text}"
        return base_prompt

    def _parse_response(self, response: str, num_choices: int, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Parse model response and add image paths."""
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

            # Save selected images if available
            if sample.has_images() and "selected_data_type" in data:
                data_type = data["selected_data_type"]
                image_path = self._save_and_get_image_path(sample, data_type)
                if image_path:
                    data["images"] = {data_type: image_path}
                else:
                    data["images"] = {}
            else:
                data["images"] = {}

            return data
        except Exception:
            return None

    def _save_and_get_image_path(self, sample: DataSample, data_type: str) -> str:
        """Save specified image type to temp folder and return path."""
        if not sample.has_images() or data_type not in sample.images:
            return ""

        image_data = sample.images[data_type]
        if not isinstance(image_data, PILImage.Image):
            return ""

        # Create output directory
        output_dir = "output/temp_images"
        os.makedirs(output_dir, exist_ok=True)

        # Generate safe filename
        safe_sample_id = sample.id.replace('/', '_').replace('\\', '_')
        safe_data_type = data_type.lower().replace('-', '_')
        filename = f"{safe_sample_id}_{safe_data_type}.png"
        file_path = os.path.join(output_dir, filename)

        # Save image
        image_data.save(file_path, "PNG")

        return file_path

    def validate_input(self, sample: DataSample) -> bool:
        """Validate input sample has required data."""
        return sample.has_text() or sample.has_images()

    def validate_output(self, output: Dict[str, Any], num_choices: int) -> bool:
        """Validate generated output format."""
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
        """Get output schema definition."""
        return {
            "type": "object",
            "properties": {
                "selected_data_type": {
                    "type": "string",
                    "description": "Selected data type used for the question.",
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
                    "description": "Selected data type and its image path",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["question", "choices", "answer"],
        }
