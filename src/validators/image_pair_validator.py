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
        return ["image_1", "image_2", "label"]

    async def validate_sample(self, sample: Dict[str, Any]) -> ValidatorResult:
        """Validate a single image pair sample"""
        # Check required fields
        issues = self._check_required_fields(sample)
        warnings = []
        suggestions = []
        metric_scores = {}

        config = self.validation_config.image_pair_config

        # Validate image files exist and are valid
        if "image_1" in sample and "image_2" in sample:
            image_validity_score = await self._validate_image_validity(sample)
            metric_scores["image_validity"] = image_validity_score

        # Validate semantic relevance
        if all(field in sample for field in ["image_1", "image_2", "label"]):
            relevance_score = await self._validate_semantic_relevance(sample)
            metric_scores["semantic_relevance"] = relevance_score

        # Validate quality consistency
        if "image_1" in sample and "image_2" in sample:
            quality_score = await self._validate_quality_consistency(sample)
            metric_scores["quality_consistency"] = quality_score

        # Calculate weighted overall score
        weights = {
            "image_validity": 2.0,  # Critical for image pairs
            "semantic_relevance": config.get("semantic_relevance_weight", 1.0),
            "quality_consistency": config.get("quality_consistency_weight", 1.0),
        }

        overall_score = self._calculate_weighted_score(metric_scores, weights)

        # Create metadata
        metadata = {
            "label": sample.get("label"),
            "image_1_path": sample.get("image_1"),
            "image_2_path": sample.get("image_2"),
            "has_caption": bool(sample.get("caption")),
            "has_metadata": bool(sample.get("metadata")),
        }

        # Add image file information if available
        for i, img_key in enumerate(["image_1", "image_2"], 1):
            img_path = sample.get(img_key)
            if img_path and os.path.exists(img_path):
                file_size = os.path.getsize(img_path)
                metadata[f"image_{i}_size_bytes"] = file_size
                metadata[f"image_{i}_extension"] = Path(img_path).suffix.lower()

        return self._create_result(
            sample=sample,
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metadata=metadata,
        )

    async def _validate_image_validity(self, sample: Dict[str, Any]) -> float:
        """Validate that image files exist and are valid"""
        score = 10.0
        config = self.validation_config.image_pair_config

        supported_formats = config.get("supported_formats", ["png", "jpg", "jpeg"])

        for img_key in ["image_1", "image_2"]:
            img_path = sample.get(img_key)

            if not img_path:
                score -= 5.0
                continue

            # Check if file exists
            if not os.path.exists(img_path):
                score -= 4.0
                continue

            # Check file extension
            file_ext = Path(img_path).suffix.lower().lstrip('.')
            if file_ext not in supported_formats:
                score -= 1.0

            # Check file size (basic validation)
            try:
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    score -= 3.0
                elif file_size < 1024:  # Very small file
                    score -= 1.0
            except OSError:
                score -= 2.0

        return max(0.0, min(10.0, score))

    async def _validate_semantic_relevance(self, sample: Dict[str, Any]) -> float:
        """Validate semantic relevance between images and label"""
        # Placeholder implementation - TODO: Implement image-semantic validation
        label = sample.get("label", "")

        if not label.strip():
            return 0.0

        score = 7.0  # Default score

        # Basic label validation
        if len(label) < 3:
            score -= 2.0
        elif len(label) > 100:
            score -= 1.0

        # Check for meaningful label content
        if label.lower() in ["positive", "negative", "similar", "different"]:
            score += 1.0

        return max(0.0, min(10.0, score))

    async def _validate_quality_consistency(self, sample: Dict[str, Any]) -> float:
        """Validate consistency in image quality"""
        # Placeholder implementation - TODO: Implement actual image quality comparison
        img1_path = sample.get("image_1")
        img2_path = sample.get("image_2")

        if not img1_path or not img2_path:
            return 0.0

        score = 8.0  # Default score

        # Basic file size comparison as quality proxy
        try:
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                size1 = os.path.getsize(img1_path)
                size2 = os.path.getsize(img2_path)

                # Check if sizes are drastically different
                if size1 > 0 and size2 > 0:
                    ratio = max(size1, size2) / min(size1, size2)
                    if ratio > 10:  # One file is 10x larger
                        score -= 2.0
                    elif ratio > 5:
                        score -= 1.0
            else:
                score -= 3.0
        except OSError:
            score -= 2.0

        return max(0.0, min(10.0, score))
