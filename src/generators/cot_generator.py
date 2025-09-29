import json
import os
import random
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample
from PIL import Image as PILImage


@register_generator("cot")
class CoTGenerator(BaseGenerator):
    """Chain-of-Thought generator for creating reasoning data with text-only and multimodal support."""

    def __init__(self, config: Dict, model_client=None):
        super().__init__(config, model_client)
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("Missing prompt_templates in config.")

        self.cot_types = config.get("cot_types", ["text_only", "multimodal"])
        self.current_type_index = 0

        # CoT-specific settings
        self.supported_spectrum_types = ["IR", "H-NMR", "C-NMR", "MASS"]
        self.image_output_dir = config.get("image_output_dir", "output/cot_spectrum_images")
        self.image_format = "png"

        # Two-stage generation settings
        self.stages = config.get("stages", ["core_generation", "reasoning_generation"])
        self.current_stage_index = 0

        # Model settings for different stages
        self.core_generation_config = config.get("core_generation", {})
        self.reasoning_generation_config = config.get("reasoning_generation", {})

        # Reasoning model settings (for future implementation)
        self.use_claude = config.get("use_claude", False)
        self.use_internvl3 = config.get("use_internvl3", True)

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single CoT reasoning sample using configurable stage approach."""
        if not self.validate_input(sample):
            return None

        if self.model_client is None:
            return None

        try:
            # Select CoT type
            cot_type = self.cot_types[self.current_type_index]

            # Check which stages to run based on configuration
            run_core_generation = "core_generation" in self.stages
            run_reasoning_generation = "reasoning_generation" in self.stages

            result = {}

            # Stage 1: Core generation (question, solution) - if enabled
            if run_core_generation:
                core_result = self._generate_core_content(sample, cot_type)
                if not core_result:
                    return None
                result.update(core_result)

            # Stage 2: Reasoning generation (thinking trajectories, attempts) - if enabled
            if run_reasoning_generation:
                reasoning_result = self._generate_reasoning_content(result, sample, cot_type)
                result.update(reasoning_result)

            if result:
                self.current_type_index = (self.current_type_index + 1) % len(self.cot_types)
                return result

            return None
        except Exception as e:
            print(f"Exception in generate_single for sample {sample.id}: {e}")
            return None

    def _prepare_model_input(self, sample: DataSample, cot_type: str, template_str: str):
        """Prepare input for the model including text prompt and optionally spectrum image."""
        prompt_text = self._build_prompt(sample, cot_type, template_str)

        if cot_type == "multimodal" and sample.has_images():
            # Select a spectrum type for multimodal
            available_spectrum_types = [
                spec_type for spec_type in sample.get_image_types() if spec_type in self.supported_spectrum_types
            ]

            if available_spectrum_types:
                selected_spectrum_type = random.choice(available_spectrum_types)
                image_path = sample.images[selected_spectrum_type]
                return {"text": prompt_text, "images": [image_path]}

        return prompt_text

    def _build_prompt(self, sample: DataSample, cot_type: str, template_str: str) -> str:
        """Build the text prompt using template and sample data."""
        # Extract molecular information
        molecule_info = sample.metadata.get("molecule_info", {})
        formula = molecule_info.get("formula", "Unknown")
        smiles = molecule_info.get("smiles", "Unknown")

        # Get available spectra types
        available_spectra = ", ".join(sample.get_image_types()) if sample.has_images() else "None"

        # Build template variables
        template_vars = {
            "formula": formula,
            "smiles": smiles,
            "available_spectra": available_spectra,
            "cot_type": cot_type,
        }

        # Add any additional metadata
        if sample.metadata:
            for key, value in sample.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    template_vars[key] = value

        try:
            prompt_template = PromptTemplate(template_str)
            formatted_prompt = prompt_template.template.format(**template_vars)
        except KeyError as e:
            formatted_prompt = template_str

        return formatted_prompt

    def _create_cot_sample(self, sample: DataSample, cot_type: str, raw_output: str) -> Optional[Dict[str, Any]]:
        """Create CoT sample from generated output."""
        try:
            # Parse the JSON response
            parsed_data = self._parse_cot_output(raw_output)
            if not parsed_data:
                return None

            # Create unique ID
            sample_id = f"cot_{sample.id}_{cot_type}"

            # Determine the type string
            type_string = f"cot {cot_type.replace('_', '-')}"

            # Create the base CoT sample
            result = {
                "id": sample_id,
                "type": type_string,
                "question": parsed_data.get("question", ""),
                "solution": parsed_data.get("solution", ""),
            }

            # Add model-specific fields based on configuration
            if self.use_claude:
                result["claude_thinking_trajectories"] = parsed_data.get("claude_thinking_trajectories", "")
                result["claude_attempt"] = parsed_data.get("claude_attempt", "")

            if self.use_internvl3:
                result["internvl3_thinking_trajectories"] = parsed_data.get("internvl3_thinking_trajectories", "")
                result["internvl3_attempt"] = parsed_data.get("internvl3_attempt", "")

            # Add image for multimodal
            if cot_type == "multimodal":
                image_path = self._save_spectrum_image(sample)
                if image_path:
                    result["image"] = image_path
                else:
                    return None

            return result

        except Exception as e:
            return None

    def _parse_cot_output(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Parse CoT output from model response."""
        try:
            # Clean the output first
            cleaned_output = raw_output.strip()

            # Try to find JSON object in the output
            start_idx = cleaned_output.find('{')
            end_idx = cleaned_output.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_output[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            # Try to parse the entire output as JSON
            if cleaned_output.startswith('{'):
                try:
                    parsed = json.loads(cleaned_output)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

            # Fallback: try to extract fields from text
            return self._extract_fields_from_text(cleaned_output)

        except Exception as e:
            print(f"Error parsing CoT output: {e}")
            return None

    def _extract_fields_from_text(self, text: str) -> Dict[str, str]:
        """Extract CoT fields from unstructured text."""
        fields = {}

        # Common field patterns
        field_patterns = {
            "question": ["question:", "question", "problem:", "problem"],
            "solution": ["solution:", "solution", "answer:", "answer"],
            "claude_thinking_trajectories": ["claude_thinking_trajectories:", "claude thinking:", "claude reasoning:"],
            "claude_attempt": ["claude_attempt:", "claude attempt:", "claude response:"],
            "internvl3_thinking_trajectories": [
                "internvl3_thinking_trajectories:",
                "internvl3 thinking:",
                "internvl3 reasoning:",
            ],
            "internvl3_attempt": ["internvl3_attempt:", "internvl3 attempt:", "internvl3 response:"],
        }

        lines = text.split('\n')
        current_field = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new field
            field_found = False
            for field_name, patterns in field_patterns.items():
                for pattern in patterns:
                    if line.lower().startswith(pattern.lower()):
                        # Save previous field
                        if current_field and current_content:
                            fields[current_field] = " ".join(current_content).strip()

                        # Start new field
                        current_field = field_name
                        content = line[len(pattern) :].strip()
                        current_content = [content] if content else []
                        field_found = True
                        break
                if field_found:
                    break

            if not field_found and current_field:
                current_content.append(line)

        # Save the last field
        if current_field and current_content:
            fields[current_field] = " ".join(current_content).strip()

        return fields

    def _save_spectrum_image(self, sample: DataSample) -> Optional[str]:
        """Save spectrum image to output directory and return path."""
        if not sample.has_images():
            return None

        # Select a random spectrum type
        available_spectrum_types = [
            spec_type for spec_type in sample.get_image_types() if spec_type in self.supported_spectrum_types
        ]

        if not available_spectrum_types:
            return None

        selected_spectrum_type = random.choice(available_spectrum_types)
        image_data = sample.images[selected_spectrum_type]

        if not isinstance(image_data, PILImage.Image):
            return None

        # Create output directory
        os.makedirs(self.image_output_dir, exist_ok=True)

        # Generate safe filename
        safe_sample_id = sample.id.replace('/', '_').replace('\\', '_')
        safe_spectrum_type = selected_spectrum_type.lower().replace('-', '_')
        filename = f"cot_{safe_sample_id}_{safe_spectrum_type}.{self.image_format}"
        file_path = os.path.join(self.image_output_dir, filename)

        try:
            image_data.save(file_path, "PNG")
            return file_path
        except Exception as e:
            return None

    def validate_input(self, sample: DataSample) -> bool:
        """Validate input sample has required data."""
        # For text-only CoT, we don't need images
        # For multimodal CoT, we need at least one spectrum image
        if "multimodal" in self.cot_types[self.current_type_index]:
            if not sample.has_images():
                return False

            available_types = sample.get_image_types()
            supported_available = [t for t in available_types if t in self.supported_spectrum_types]
            return len(supported_available) > 0

        return True

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate generated output format."""
        if not super().validate_output(output):
            return False

        required_fields = ["id", "type", "question", "solution"]
        if not all(field in output for field in required_fields):
            return False

        # Check for at least one model's output
        has_claude = "claude_thinking_trajectories" in output or "claude_attempt" in output
        has_internvl3 = "internvl3_thinking_trajectories" in output or "internvl3_attempt" in output

        if not (has_claude or has_internvl3):
            return False

        # For multimodal, check image field
        if output.get("type") == "cot multimodal":
            if "image" not in output:
                return False

        return True

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema definition."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for the CoT sample"},
                "type": {
                    "type": "string",
                    "enum": ["cot text-only", "cot multimodal"],
                    "description": "Data type identifier",
                },
                "question": {"type": "string", "description": "The reasoning question or problem"},
                "solution": {"type": "string", "description": "The final solution or answer"},
            },
            "required": ["id", "type", "question", "solution"],
        }

        # Add optional fields based on model configuration
        if self.use_claude:
            schema["properties"]["claude_thinking_trajectories"] = {
                "type": "string",
                "description": "Claude's thinking process",
            }
            schema["properties"]["claude_attempt"] = {"type": "string", "description": "Claude's attempt at solving"}

        if self.use_internvl3:
            schema["properties"]["internvl3_thinking_trajectories"] = {
                "type": "string",
                "description": "InternVL3's thinking process",
            }
            schema["properties"]["internvl3_attempt"] = {
                "type": "string",
                "description": "InternVL3's attempt at solving",
            }

        # Add image field for multimodal
        schema["properties"]["image"] = {
            "type": "string",
            "description": "Path to the spectrum image file (multimodal only)",
        }

        return schema

    def _generate_core_content(self, sample: DataSample, cot_type: str) -> Optional[Dict[str, Any]]:
        """Stage 1: Generate core content (id, type, question, solution)."""
        try:
            # Get core generation template
            core_template = self.prompt_templates.get("core_generation", {}).get(cot_type)
            if not core_template:
                # Fallback to old template structure
                core_template = self.prompt_templates.get(cot_type)

            if isinstance(core_template, dict):
                template_str = core_template["template"]
            else:
                template_str = core_template

            # Prepare model input
            model_input = self._prepare_model_input(sample, cot_type, template_str)

            # Generate with core generation settings
            max_out_len = self.core_generation_config.get("max_out_len", 2000)
            raw_output = self.model_client.generate(model_input, max_out_len=max_out_len)

            if raw_output:
                return self._parse_core_content(sample, cot_type, raw_output)

            return None
        except Exception as e:
            print(f"Error in core generation for sample {sample.id}: {e}")
            return None

    def _generate_reasoning_content(
        self, core_result: Dict[str, Any], sample: DataSample, cot_type: str
    ) -> Dict[str, Any]:
        """Stage 2: Generate reasoning content (thinking trajectories, attempts)."""
        # For now, return empty reasoning content since reasoning models aren't ready
        # In the future, this would use reasoning models to generate thinking trajectories and attempts
        reasoning_result = {}

        # Placeholder reasoning content (will be replaced with actual reasoning model calls)
        if self.use_claude:
            reasoning_result["claude_thinking_trajectories"] = ""
            reasoning_result["claude_attempt"] = ""

        if self.use_internvl3:
            reasoning_result["internvl3_thinking_trajectories"] = ""
            reasoning_result["internvl3_attempt"] = ""

        return reasoning_result

    def _parse_core_content(self, sample: DataSample, cot_type: str, raw_output: str) -> Optional[Dict[str, Any]]:
        """Parse core content from model output."""
        try:
            # Parse the JSON response
            parsed_data = self._parse_cot_output(raw_output)
            if not parsed_data:
                return None

            # Create unique ID
            sample_id = f"cot_{sample.id}_{cot_type}"

            # Determine the type string
            type_string = f"cot {cot_type.replace('_', '-')}"

            # Create the base CoT sample with core content only
            result = {
                "id": sample_id,
                "type": type_string,
                "question": parsed_data.get("question", ""),
                "solution": parsed_data.get("solution", ""),
            }

            # Add image for multimodal
            if cot_type == "multimodal":
                image_path = self._save_spectrum_image(sample)
                if image_path:
                    result["image"] = image_path

            return result

        except Exception as e:
            print(f"Error parsing core content: {e}")
            return None
