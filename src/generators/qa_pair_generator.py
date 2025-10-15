import json
import os
import random
from typing import Dict, Any, Optional, List
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample
from PIL import Image as PILImage


@register_generator("qa_pair")
class QAPairGenerator(BaseGenerator):
    """QA pair generator for creating single-step and multi-step question-answer pairs."""

    def __init__(self, config: Dict, model_client=None):
        super().__init__(config, model_client)
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("Missing prompt_templates in config.")

        self.qa_types = config.get("qa_types", ["single_step", "multi_step"])
        self.current_type_index = 0

        # QA-specific settings
        self.min_conversation_length = config.get("min_conversation_length", 2)
        self.max_conversation_length = config.get("max_conversation_length", 6)
        # Note: Supported spectrum types are now automatically detected from DataSample
        # No hardcoded spectrum types - works with any image types in the data
        self.image_output_dir = config.get("image_output_dir", "output/qa_spectrum_images")
        self.image_format = "png"

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single QA pair."""
        if not self.validate_input(sample):
            return None

        if self.model_client is None:
            return None

        try:
            # Get all available spectrum types from the sample (dataset-agnostic)
            available_spectrum_types = sample.get_image_types()

            if not available_spectrum_types:
                return None

            selected_spectrum_type = random.choice(available_spectrum_types)

            # Select QA type
            qa_type = self.qa_types[self.current_type_index]
            template_info = self.prompt_templates[qa_type]

            if isinstance(template_info, dict):
                template_str = template_info["template"]
            else:
                template_str = template_info

            # Prepare model input
            model_input = self._prepare_model_input(sample, selected_spectrum_type, template_str)

            try:
                max_out_len = self.config.get("max_out_len", 2000)
                raw_output = self.model_client.generate(model_input, max_out_len=max_out_len)
                if raw_output:
                    result = self._create_qa_pair(sample, selected_spectrum_type, qa_type, raw_output)
                    if result:
                        self.current_type_index = (self.current_type_index + 1) % len(self.qa_types)
                        return result
            except Exception as e:
                print(f"Exception in generate_single for sample {sample.id}: {e}")

            return None
        except Exception as e:
            return None

    def _prepare_model_input(self, sample: DataSample, spectrum_type: str, template_str: str):
        """Prepare input for the model including text prompt and spectrum image."""
        prompt_text = self._build_prompt(sample, spectrum_type, template_str)

        if sample.has_images() and spectrum_type in sample.images:
            image_path = sample.images[spectrum_type]
            return {"text": prompt_text, "images": [image_path]}
        else:
            return prompt_text

    def _build_prompt(self, sample: DataSample, spectrum_type: str, template_str: str) -> str:
        """Build the text prompt using template and sample data."""
        # Extract molecular information
        molecule_info = sample.metadata.get("molecule_info", {})
        formula = molecule_info.get("formula", "Unknown")
        smiles = molecule_info.get("smiles", "Unknown")

        # Get available spectra types
        available_spectra = ", ".join(sample.get_image_types()) if sample.has_images() else "None"

        # Build template variables
        template_vars = {
            "spectrum_type": spectrum_type,
            "formula": formula,
            "smiles": smiles,
            "available_spectra": available_spectra,
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

    def _create_qa_pair(
        self, sample: DataSample, spectrum_type: str, qa_type: str, raw_output: str
    ) -> Optional[Dict[str, Any]]:
        """Create QA pair from generated output."""
        try:
            # Parse the JSON response
            conversations = self._parse_conversations(raw_output)
            if not conversations:
                return None

            # Replace template variables in conversations
            conversations = self._replace_template_variables(conversations, sample, spectrum_type)

            # Limit conversations based on QA type
            conversations = self._limit_conversations(conversations, qa_type)

            # Save the spectrum image
            image_path = self._save_spectrum_image(sample, spectrum_type)
            if not image_path:
                return None

            # Create unique ID
            pair_id = f"qa_{sample.id}_{spectrum_type.lower().replace('-', '_')}_{qa_type}"

            # Determine the type string
            type_string = f"QA pair/{qa_type.replace('_', '-')}"

            # Create the QA pair
            result = {"id": pair_id, "type": type_string, "image": image_path, "conversations": conversations}

            return result

        except Exception as e:
            return None

    def _parse_conversations(self, raw_output: str) -> Optional[List[Dict[str, str]]]:
        """Parse conversations from model output."""
        try:
            # Clean the output first
            cleaned_output = raw_output.strip()

            # Try to find JSON array in the output
            start_idx = cleaned_output.find('[')
            end_idx = cleaned_output.rfind(']')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = cleaned_output[start_idx : end_idx + 1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass

            # Try to parse the entire output as JSON
            if cleaned_output.startswith('{') or cleaned_output.startswith('['):
                try:
                    parsed = json.loads(cleaned_output)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "conversations" in parsed:
                        return parsed["conversations"]
                except json.JSONDecodeError:
                    pass

            # Fallback: try to extract conversations from text
            conversations = []
            lines = cleaned_output.split('\n')

            current_speaker = None
            current_content = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for speaker indicators
                if '"from":' in line or 'from:' in line:
                    if current_speaker and current_content:
                        conversations.append({"from": current_speaker, "value": " ".join(current_content).strip()})
                    # Extract speaker
                    if '"from":' in line:
                        current_speaker = line.split('"from":')[1].split(',')[0].strip().strip('"\'')
                    else:
                        current_speaker = line.split('from:')[1].strip().strip('"\'')
                    current_content = []
                elif '"value":' in line or 'value:' in line:
                    # Extract value
                    if '"value":' in line:
                        content = line.split('"value":')[1].strip().strip('"\'')
                    else:
                        content = line.split('value:')[1].strip().strip('"\'')
                    if content:
                        current_content.append(content)
                elif current_speaker and line and not line.startswith('{') and not line.startswith('}'):
                    current_content.append(line)

            # Add the last conversation
            if current_speaker and current_content:
                conversations.append({"from": current_speaker, "value": " ".join(current_content).strip()})

            return conversations if conversations else None

        except Exception as e:
            print(f"Error parsing conversations: {e}")
            return None

    def _replace_template_variables(
        self, conversations: List[Dict[str, str]], sample: DataSample, spectrum_type: str
    ) -> List[Dict[str, str]]:
        """Replace template variables in conversations with actual values."""
        # Extract molecular information
        molecule_info = sample.metadata.get("molecule_info", {})
        formula = molecule_info.get("formula", "Unknown")
        smiles = molecule_info.get("smiles", "Unknown")

        # Create replacement mapping
        replacements = {
            "{formula}": formula,
            "{smiles}": smiles,
            "{spectrum_type}": spectrum_type,
        }

        # Replace variables in each conversation
        updated_conversations = []
        for conv in conversations:
            updated_conv = conv.copy()
            value = conv.get("value", "")

            # Replace all template variables
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)

            updated_conv["value"] = value
            updated_conversations.append(updated_conv)

        return updated_conversations

    def _limit_conversations(self, conversations: List[Dict[str, str]], qa_type: str) -> List[Dict[str, str]]:
        """Limit conversations based on QA type."""
        if qa_type == "single_step":
            # Single step should have exactly 2 conversations (1 Q + 1 A)
            return conversations[:2] if len(conversations) >= 2 else conversations
        elif qa_type == "multi_step":
            # Multi step should have 4-6 conversations (2-3 Q + 2-3 A)
            return conversations[:6] if len(conversations) >= 4 else conversations
        else:
            return conversations

    def _save_spectrum_image(self, sample: DataSample, spectrum_type: str) -> Optional[str]:
        """Save spectrum image to output directory and return path."""
        if not sample.has_images() or spectrum_type not in sample.images:
            return None

        image_data = sample.images[spectrum_type]
        if not isinstance(image_data, PILImage.Image):
            return None

        # Create output directory
        os.makedirs(self.image_output_dir, exist_ok=True)

        # Generate safe filename
        safe_sample_id = sample.id.replace('/', '_').replace('\\', '_')
        safe_spectrum_type = spectrum_type.lower().replace('-', '_')
        filename = f"qa_{safe_sample_id}_{safe_spectrum_type}.{self.image_format}"
        file_path = os.path.join(self.image_output_dir, filename)

        try:
            image_data.save(file_path, "PNG")
            return file_path
        except Exception as e:
            return None

    def validate_input(self, sample: DataSample) -> bool:
        """Validate input sample has required data."""
        # Simply check if sample has any images (dataset-agnostic)
        return sample.has_images()

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate generated output format."""
        if not super().validate_output(output):
            return False

        required_fields = ["id", "type", "image", "conversations"]
        if not all(field in output for field in required_fields):
            return False

        # Validate conversations structure
        conversations = output.get("conversations", [])
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False

        for conv in conversations:
            if not isinstance(conv, dict) or "from" not in conv or "value" not in conv:
                return False
            if conv["from"] not in ["human", "gpt"]:
                return False

        return True

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema definition."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for the QA pair"},
                "type": {
                    "type": "string",
                    "enum": ["QA pair/single-step", "QA pair/multi-step"],
                    "description": "Data type identifier",
                },
                "image": {"type": "string", "description": "Path to the spectrum image file"},
                "conversations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "enum": ["human", "gpt"]},
                            "value": {"type": "string"},
                        },
                        "required": ["from", "value"],
                    },
                    "minItems": 2,
                },
            },
            "required": ["id", "type", "image", "conversations"],
        }
