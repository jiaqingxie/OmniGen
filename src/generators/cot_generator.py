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
        # Note: Supported spectrum types are now automatically detected from DataSample
        # No hardcoded spectrum types - works with any image types in the data
        self.image_output_dir = config.get("image_output_dir", "output/cot_spectrum_images")
        self.image_format = "png"

        # Two-stage generation settings
        self.stages = config.get("stages", ["draft", "reason"])
        self.current_stage_index = 0

        # Model settings for different stages
        self.draft_config = config.get("draft", {})
        self.reason_config = config.get("reason", {})

        # Reasoning model settings (for future implementation)
        self.use_claude = config.get("use_claude", False)
        self.use_interns1 = config.get("use_interns1", True)

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
            run_draft = "draft" in self.stages
            run_reason = "reason" in self.stages

            # Special handling for reason-only generation
            if run_reason and not run_draft:
                # Allow reason-only generation for incremental processing
                # We'll use existing content from the sample if available
                pass

            result = {}

            # Stage 1: Draft generation (question, solution) - if enabled
            if run_draft:
                draft_result = self._generate_draft(sample, cot_type)
                if not draft_result:
                    return None
                result.update(draft_result)

            # Stage 2: Reason generation (thinking trajectories, attempts) - if enabled
            if run_reason:
                # For reason-only mode, pass empty result if no draft was generated
                content_for_reason = result if result else {}
                reason_result = self._generate_reason(content_for_reason, sample, cot_type)
                result.update(reason_result)

            if result:
                self.current_type_index = (self.current_type_index + 1) % len(self.cot_types)
                return result
            else:
                print(f"Debug - No result generated for sample {sample.id}")
                return None
        except Exception as e:
            print(f"Exception in generate_single for sample {sample.id}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _prepare_model_input(self, sample: DataSample, cot_type: str, template_str: str):
        """Prepare input for the model including text prompt and optionally spectrum image."""
        selected_spectrum_type = None

        if cot_type == "multimodal" and sample.has_images():
            # Select a spectrum type for multimodal
            # Use all available spectrum types from the sample (dataset-agnostic)
            available_spectrum_types = sample.get_image_types()

            if available_spectrum_types:
                selected_spectrum_type = random.choice(available_spectrum_types)
                # Save to metadata for later use in _save_spectrum_image
                sample.metadata["_selected_spectrum_type"] = selected_spectrum_type

        # Build prompt with selected spectrum type
        prompt_text = self._build_prompt(sample, cot_type, template_str, selected_spectrum_type)

        if selected_spectrum_type:
            image_path = sample.images[selected_spectrum_type]
            return {"text": prompt_text, "images": [image_path]}

        return prompt_text

    def _build_prompt(
        self, sample: DataSample, cot_type: str, template_str: str, selected_spectrum_type: Optional[str] = None
    ) -> str:
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
            "spectrum_type": selected_spectrum_type if selected_spectrum_type else "N/A",
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
            extracted_fields = self._extract_fields_from_text(cleaned_output)
            if extracted_fields:
                return extracted_fields
            else:
                return None

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
            "interns1_thinking_trajectories": [
                "interns1_thinking_trajectories:",
                "interns1 thinking:",
                "interns1 reasoning:",
            ],
            "interns1_attempt": ["interns1_attempt:", "interns1 attempt:", "interns1 response:"],
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
        filename = f"cot_{safe_sample_id}_{safe_spectrum_type}.{self.image_format}"
        file_path = os.path.join(self.image_output_dir, filename)

        try:
            image_data.save(file_path, "PNG")
            return file_path
        except Exception as e:
            return None

    def validate_input(self, sample: DataSample) -> bool:
        """Validate input sample has required data."""
        # For reason-only mode, we only need existing content in metadata
        if "reason" in self.stages and "draft" not in self.stages:
            # Check if we have existing content in metadata for incremental generation
            has_existing_content = sample.metadata.get("existing_question") or sample.metadata.get("existing_solution")
            if has_existing_content:
                return True
            # If no existing content, fall through to normal validation

        # For text-only CoT, we don't need images
        # For multimodal CoT, we need at least one image
        if "multimodal" in self.cot_types[self.current_type_index]:
            return sample.has_images()

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
        has_interns1 = "interns1_thinking_trajectories" in output or "interns1_attempt" in output

        if not (has_claude or has_interns1):
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

        if self.use_interns1:
            schema["properties"]["interns1_thinking_trajectories"] = {
                "type": "string",
                "description": "InternS1's thinking process",
            }
            schema["properties"]["interns1_attempt"] = {
                "type": "string",
                "description": "InternS1's attempt at solving",
            }

        # Add image field for multimodal
        schema["properties"]["image"] = {
            "type": "string",
            "description": "Path to the spectrum image file (multimodal only)",
        }

        return schema

    def _generate_draft(self, sample: DataSample, cot_type: str) -> Optional[Dict[str, Any]]:
        """Stage 1: Generate draft content (id, type, question, solution)."""
        try:
            # Get draft template
            draft_template = self.prompt_templates.get("draft", {}).get(cot_type)
            if not draft_template:
                # Fallback to old template structure
                draft_template = self.prompt_templates.get(cot_type)

            if isinstance(draft_template, dict):
                template_str = draft_template["template"]
            else:
                template_str = draft_template

            # Prepare model input
            model_input = self._prepare_model_input(sample, cot_type, template_str)

            # Generate with draft settings
            max_out_len = self.draft_config.get("max_out_len", 8000)
            raw_output = self.model_client.generate(model_input, max_out_len=max_out_len)

            if raw_output:
                return self._parse_draft(sample, cot_type, raw_output)

            return None
        except Exception as e:
            print(f"Error in content generation for sample {sample.id}: {e}")
            return None

    def _generate_reason(self, content_result: Dict[str, Any], sample: DataSample, cot_type: str) -> Dict[str, Any]:
        """Stage 2: Generate reason content using InternS1 model."""
        reasoning_result = {}

        try:
            # Use InternS1 for reasoning generation
            if hasattr(self.model_client, 'generate_with_reasoning'):
                # InternS1 model - get both content and reasoning
                question = content_result.get("question", "")
                solution = content_result.get("solution", "")

                # If no content from previous stage, try to get from sample metadata (for incremental generation)
                if not question and not solution:
                    question = sample.metadata.get("existing_question", "")
                    solution = sample.metadata.get("existing_solution", "")

                # If still no content, create a generic reasoning prompt
                if not question and not solution:
                    # Create a generic reasoning prompt based on sample data
                    molecule_info = sample.metadata.get("molecule_info", {})
                    formula = molecule_info.get("formula", "Unknown")
                    smiles = molecule_info.get("smiles", "Unknown")

                    reasoning_prompt = f"""
                    Generate detailed reasoning trajectories and model attempts for a spectroscopic analysis problem.
                    
                    Molecular Information:
                    - Formula: {formula}
                    - SMILES: {smiles}
                    
                    Please provide:
                    1. Thinking trajectories (step-by-step reasoning process for spectroscopic analysis)
                    2. Model attempt (final answer attempt for the molecular structure)
                    """
                else:
                    # Create reasoning prompt based on existing content
                    reasoning_prompt = f"""
                    Based on the following question and solution, generate detailed reasoning trajectories and model attempts.
                    
                    Question: {question}
                    Solution: {solution}
                    
                    Please provide:
                    1. Thinking trajectories (step-by-step reasoning process)
                    2. Model attempt (final answer attempt)
                    """

                # Generate with InternS1
                max_out_len = self.reason_config.get("max_out_len", 8000)  # 增加token限制
                reasoning_response = self.model_client.generate_with_reasoning(
                    reasoning_prompt, max_out_len=max_out_len
                )

                # Extract reasoning content
                reasoning_content = reasoning_response.get("reasoning_content", "")
                model_attempt = reasoning_response.get("content", "")

                # Store in appropriate fields
                if self.use_interns1:
                    reasoning_result["interns1_thinking_trajectories"] = reasoning_content
                    reasoning_result["interns1_attempt"] = model_attempt

                if self.use_claude:
                    # For now, use the same content for Claude (can be replaced with actual Claude model later)
                    reasoning_result["claude_thinking_trajectories"] = reasoning_content
                    reasoning_result["claude_attempt"] = model_attempt

                # For incremental generation, preserve original content
                if not content_result and sample.metadata.get("original_data"):
                    original_data = sample.metadata["original_data"]
                    reasoning_result["id"] = original_data.get("id", sample.id)
                    reasoning_result["type"] = original_data.get("type", f"cot {cot_type}")
                    reasoning_result["question"] = original_data.get("question", "")
                    reasoning_result["solution"] = original_data.get("solution", "")

            else:
                # Fallback for non-InternS1 models
                reasoning_result = self._generate_reason_fallback(content_result, sample, cot_type)

        except Exception as e:
            print(f"Error in reasoning generation for sample {sample.id}: {e}")
            # Return empty reasoning content on error
            reasoning_result = self._generate_reason_fallback(content_result, sample, cot_type)

        return reasoning_result

    def _generate_reason_fallback(
        self, content_result: Dict[str, Any], sample: DataSample, cot_type: str
    ) -> Dict[str, Any]:
        """Fallback reason generation when InternS1 is not available."""
        reasoning_result = {}

        # Placeholder reasoning content
        if self.use_claude:
            reasoning_result["claude_thinking_trajectories"] = ""
            reasoning_result["claude_attempt"] = ""

        if self.use_interns1:
            reasoning_result["interns1_thinking_trajectories"] = ""
            reasoning_result["interns1_attempt"] = ""

        return reasoning_result

    def _parse_draft(self, sample: DataSample, cot_type: str, raw_output: str) -> Optional[Dict[str, Any]]:
        """Parse draft content from model output."""
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
                # Get the spectrum type from metadata (saved during _prepare_model_input)
                spectrum_type = sample.metadata.get("_selected_spectrum_type")
                if spectrum_type:
                    image_path = self._save_spectrum_image(sample, spectrum_type)
                    if image_path:
                        result["image"] = image_path

            return result

        except Exception as e:
            print(f"Error parsing core content: {e}")
            return None
