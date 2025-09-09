from typing import Dict, Any, List
import os
from pathlib import Path
from .base import BaseValidator
from .registry import register_validator
from .result import ValidatorResult


@register_validator("image_pair")
class ImagePairValidator(BaseValidator):
    """Validator for image pair data"""

    def get_supported_data_type(self) -> str:
        return "image_pair"

    def get_required_fields(self) -> List[str]:
        return ["id", "type", "image", "text"]

    async def validate_sample(self, sample: Dict[str, Any]) -> ValidatorResult:
        """Validate a single image-text pair sample"""
        # Check required fields
        issues = self._check_required_fields(sample)
        warnings = []
        suggestions = []
        metric_scores = {}

        config = self.validation_config.image_pair_config

        # Validate data type field
        if sample.get("type") != "image-text pair":
            issues.append("Invalid type field - must be 'image-text pair'")

        # Validate image file exists and is valid
        if "image" in sample:
            image_validity_score = await self._validate_image_validity(sample)
            metric_scores["image_validity"] = image_validity_score

        # Validate text quality
        if "text" in sample:
            text_quality_score = await self._validate_text_quality(sample)
            metric_scores["text_quality"] = text_quality_score

        # Validate image-text relevance
        if all(field in sample for field in ["image", "text"]):
            relevance_score = await self._validate_image_text_relevance(sample)
            metric_scores["image_text_relevance"] = relevance_score

        # Calculate weighted overall score
        weights = {
            "image_validity": 2.0,  # Critical for image-text pairs
            "text_quality": config.get("text_quality_weight", 1.5),
            "image_text_relevance": config.get("semantic_relevance_weight", 2.0),
        }

        overall_score = self._calculate_weighted_score(metric_scores, weights)

        # Create metadata
        metadata = {
            "sample_id": sample.get("id"),
            "data_type": sample.get("type"),
            "image_path": sample.get("image"),
            "text_length": len(sample.get("text", "")),
            "has_valid_id": bool(sample.get("id", "").strip()),
        }

        # Add image file information if available
        img_path = sample.get("image")
        if img_path and os.path.exists(img_path):
            try:
                file_size = os.path.getsize(img_path)
                metadata["image_size_bytes"] = file_size
                metadata["image_extension"] = Path(img_path).suffix.lower()
            except OSError:
                metadata["image_size_bytes"] = 0

        # Create custom result for image-pair format (uses "id" instead of "sample_id")
        is_valid = overall_score >= self.validation_config.overall_threshold and len(issues) == 0

        return ValidatorResult(
            sample_id=sample.get("id", "unknown"),  # Use "id" field for image-pair data
            data_type=self.get_supported_data_type(),
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=issues,
            warnings=warnings or [],
            suggestions=suggestions or [],
            is_valid=is_valid,
            metadata=metadata or {},
        )

    async def _validate_image_validity(self, sample: Dict[str, Any]) -> float:
        """Validate that image file exists and is valid"""
        score = 10.0
        config = self.validation_config.image_pair_config

        supported_formats = config.get("supported_formats", ["png", "jpg", "jpeg"])
        img_path = sample.get("image")

        if not img_path:
            return 0.0

        # Check if file exists
        if not os.path.exists(img_path):
            score -= 5.0
        else:
            # Check file extension
            file_ext = Path(img_path).suffix.lower().lstrip('.')
            if file_ext not in supported_formats:
                score -= 2.0

            # Check file size (basic validation)
            try:
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    score -= 4.0
                elif file_size < 1024:  # Very small file
                    score -= 1.0
            except OSError:
                score -= 3.0

        return max(0.0, min(10.0, score))

    async def _validate_text_quality(self, sample: Dict[str, Any]) -> float:
        """Validate quality of the descriptive text"""
        text = sample.get("text", "")

        if not text.strip():
            return 0.0

        score = 8.0  # Default score
        config = self.validation_config.image_pair_config

        # Length validation
        text_length = len(text)
        min_length = config.get("min_text_length", 50)
        max_length = config.get("max_text_length", 300)

        if text_length < min_length:
            score -= 3.0
        elif text_length > max_length:
            score -= 2.0

        # Content quality checks
        text_lower = text.lower()

        # Check for spectrum-related keywords
        spectrum_keywords = ["spectrum", "peak", "absorption", "chemical", "molecule", "formula"]
        keyword_count = sum(1 for keyword in spectrum_keywords if keyword in text_lower)

        if keyword_count >= 3:
            score += 1.0
        elif keyword_count < 2:
            score -= 2.0

        # Check for scientific language indicators
        scientific_indicators = ["ppm", "cm⁻¹", "nmr", "infrared", "mass spec", "smiles"]
        if any(indicator in text_lower for indicator in scientific_indicators):
            score += 0.5

        # Basic grammar checks
        if not text[0].isupper():
            score -= 0.5

        if not text.rstrip().endswith(('.', '!', '?')):
            score -= 0.5

        return max(0.0, min(10.0, score))

    async def _validate_image_text_relevance(self, sample: Dict[str, Any]) -> float:
        """Validate relevance between image and text description"""
        # Placeholder implementation - TODO: Implement multimodal relevance validation
        text = sample.get("text", "").lower()
        img_path = sample.get("image", "")

        if not text.strip() or not img_path:
            return 0.0

        score = 7.0  # Default score

        # Extract spectrum type from image path if possible
        img_name = Path(img_path).stem.lower()
        spectrum_types = ["ir", "nmr", "h_nmr", "c_nmr", "mass", "ms"]

        detected_spectrum = None
        for spec_type in spectrum_types:
            if spec_type in img_name:
                detected_spectrum = spec_type
                break

        # Check if text mentions the correct spectrum type
        if detected_spectrum:
            spectrum_mentions = {
                "ir": ["infrared", "ir"],
                "nmr": ["nmr", "nuclear magnetic"],
                "h_nmr": ["proton", "1h", "h-nmr", "hnmr"],
                "c_nmr": ["carbon", "13c", "c-nmr", "cnmr"],
                "mass": ["mass", "ms", "spectrometry"],
                "ms": ["mass", "ms", "spectrometry"],
            }

            relevant_terms = spectrum_mentions.get(detected_spectrum, [])
            if any(term in text for term in relevant_terms):
                score += 1.5
            else:
                score -= 1.0

        # Check for molecular information consistency
        if "formula" in text and ("smiles" in text or "chemical" in text):
            score += 0.5

        return max(0.0, min(10.0, score))
