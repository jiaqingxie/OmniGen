import os
import random
from typing import Dict, Any, Optional, Tuple
from .base import BaseGenerator, PromptTemplate, register_generator
from ..data_loaders import DataSample
from PIL import Image as PILImage


@register_generator("image_pair")
class ImagePairGenerator(BaseGenerator):
    """Image-text pair generator for creating spectrum descriptions."""

    def __init__(self, config: Dict, model_client=None):
        super().__init__(config, model_client)
        self.prompt_templates = config.get("prompt_templates", {})
        if not self.prompt_templates:
            raise ValueError("Missing prompt_templates in config.")

        self.description_types = config.get("description_types", list(self.prompt_templates.keys()))
        self.current_type_index = 0

        # Image-pair specific settings
        self.min_text_length = config.get("min_text_length", 50)
        self.max_text_length = config.get("max_text_length", 2500)
        # Note: Supported spectrum types are now automatically detected from DataSample
        # No hardcoded spectrum types - the generator works with any image types in the data
        self.image_output_dir = config.get("image_output_dir", "output/spectrum_images")
        self.image_format = "png"  # Default format
        self.default_description_type = config.get("default_description_type", "basic_description")

    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single image-text pair."""
        if not self.validate_input(sample):
            return None

        # Handle missing model client gracefully
        if self.model_client is None:
            print("[ImagePairGenerator] Missing model client; cannot generate output.")
            return None

        # Get all available spectrum types from the sample
        # This is dataset-agnostic and works with any spectrum types
        available_spectrum_types = sample.get_image_types()

        if not available_spectrum_types:
            print(f"[ImagePairGenerator] Sample {sample.id} contains no spectrum images.")
            return None

        selected_spectrum_type = random.choice(available_spectrum_types)

        # Select description type
        description_type = self.description_types[self.current_type_index]
        template_info = self.prompt_templates[description_type]

        if isinstance(template_info, dict):
            template_str = template_info["template"]
        else:
            template_str = template_info

        # Prepare model input
        model_input = self._prepare_model_input(sample, selected_spectrum_type, template_str)

        # Generate description using LLM
        max_out_len = self.config.get("max_out_len", 1500)
        try:
            raw_output = self.model_client.generate(model_input, max_out_len=max_out_len)
        except Exception as exc:
            print(f"[ImagePairGenerator] Model generation failed for sample {sample.id}: {exc}")
            return None

        if not raw_output:
            print(f"[ImagePairGenerator] Empty response received for sample {sample.id}.")
            return None

        result = self._create_image_pair(sample, selected_spectrum_type, raw_output)
        if result:
            self.current_type_index = (self.current_type_index + 1) % len(self.description_types)
        return result

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

        # Derive spectrum-specific auxiliary fields (e.g., chemical shifts) from molecule info
        spectrum_upper = (spectrum_type or "").upper()
        chemical_shifts = None
        if spectrum_upper == "H-NMR":
            chemical_shifts = molecule_info.get("h_nmr_chemical_shift")
        elif spectrum_upper == "C-NMR":
            chemical_shifts = molecule_info.get("c_nmr_chemical_shift")

        # Get available spectra types
        available_spectra = ", ".join(sample.get_image_types()) if sample.has_images() else "None"

        # Build template variables
        template_vars = {
            "spectrum_type": spectrum_type,
            "formula": formula,
            "smiles": smiles,
            "available_spectra": available_spectra,
            "chemical_shifts": chemical_shifts if chemical_shifts is not None else "Unknown",
        }

        # Add any additional metadata
        if sample.metadata:
            for key, value in sample.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    template_vars[key] = value

        # Use safe formatting so missing keys don't drop already-resolved fields
        class _DefaultDict(dict):
            def __missing__(self, key):
                return "Unknown"

        prompt_template = PromptTemplate(template_str)
        formatted_prompt = prompt_template.template.format_map(_DefaultDict(template_vars))

        return formatted_prompt

    def _create_image_pair(
        self, sample: DataSample, spectrum_type: str, raw_description: str
    ) -> Optional[Dict[str, Any]]:
        """Create image-text pair from generated description."""
        # Clean and validate the description
        description = self._clean_description(raw_description)

        is_valid, error_message = self._validate_description(description)
        if not is_valid:
            detail = f": {error_message}" if error_message else ""
            print(f"[ImagePairGenerator] Validation failed for sample {sample.id}{detail}")
            return None

        # Save the spectrum image
        image_path = self._save_spectrum_image(sample, spectrum_type)
        if not image_path:
            print(f"[ImagePairGenerator] Failed to save image for sample {sample.id} ({spectrum_type}).")
            return None

        # Create unique ID
        pair_id = f"{sample.id}_{spectrum_type.lower().replace('-', '_')}"

        # Create the image-text pair
        result = {"id": pair_id, "type": "image-text pair", "image": image_path, "text": description}

        return result

    def _clean_description(self, raw_description: str) -> str:
        """Clean and process the generated description."""
        # Remove extra whitespace and newlines
        description = " ".join(raw_description.strip().split())

        # Remove any quotation marks that might have been added
        description = description.strip('"\'')

        # Ensure proper capitalization
        if description and not description[0].isupper():
            description = description[0].upper() + description[1:]

        # Ensure proper ending punctuation
        if description and description[-1] not in '.!?':
            description += '.'

        return description

    def _validate_description(self, description: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated description."""
        if not description or not description.strip():
            return False, "description is empty"

        # Check length constraints
        desc_len = len(description)
        if desc_len < self.min_text_length:
            return False, f"description too short ({desc_len} < {self.min_text_length})"

        if desc_len > self.max_text_length:
            return False, f"description too long ({desc_len} > {self.max_text_length})"

        # Basic quality checks
        # Check if it contains key information
        required_keywords = ["spectrum", "molecule", "formula"]
        description_lower = description.lower()

        keyword_count = sum(1 for keyword in required_keywords if keyword in description_lower)
        if keyword_count < 2:  # At least 2 out of 3 keywords should be present
            return False, f"insufficient keywords ({keyword_count}/3 required keywords present)"

        return True, None

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
        filename = f"{safe_sample_id}_{safe_spectrum_type}.{self.image_format}"
        file_path = os.path.join(self.image_output_dir, filename)

        try:
            # Save image as PNG (default format)
            image_data.save(file_path, "PNG")
            return file_path
        except Exception as exc:
            print(f"[ImagePairGenerator] Failed to write image {file_path}: {exc}")
            return None

    def validate_input(self, sample: DataSample) -> bool:
        """Validate input sample has required data."""
        # Simply check if sample has any images
        # No restriction on spectrum types - works with any image types
        return sample.has_images()

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate generated output format."""
        if not super().validate_output(output):
            return False

        required_fields = ["id", "type", "image", "text"]
        if not all(field in output for field in required_fields):
            return False

        # Validate specific field values
        if output["type"] != "image-text pair":
            return False

        # Check if image file exists
        if not os.path.exists(output["image"]):
            return False

        # Validate text
        if not isinstance(output["text"], str) or not output["text"].strip():
            return False

        return True

    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema definition."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for the image-text pair"},
                "type": {"type": "string", "enum": ["image-text pair"], "description": "Data type identifier"},
                "image": {"type": "string", "description": "Path to the spectrum image file"},
                "text": {"type": "string", "description": "Descriptive text for the spectrum image"},
            },
            "required": ["id", "type", "image", "text"],
        }
