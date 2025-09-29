from typing import Dict, Any, List
from .base import BaseValidator
from .registry import register_validator
from .result import ValidatorResult


@register_validator("qa_pair")
class QAPairValidator(BaseValidator):
    """Validator for QA pair data"""

    def get_supported_data_type(self) -> str:
        return "qa_pair"

    def get_required_fields(self) -> List[str]:
        return ["id", "type", "image", "conversations"]

    async def validate_sample(self, sample: Dict[str, Any]) -> ValidatorResult:
        """Validate a single QA pair sample"""
        # Check required fields
        issues = self._check_required_fields(sample)
        warnings = []
        suggestions = []
        metric_scores = {}

        config = self.validation_config.qa_pair_config

        # Validate conversation structure
        if "conversations" in sample:
            conversation_score = await self._validate_conversation_structure(sample)
            metric_scores["conversation_structure"] = conversation_score

        # Validate conversation quality
        if "conversations" in sample:
            quality_score = await self._validate_conversation_quality(sample)
            metric_scores["conversation_quality"] = quality_score

        # Validate image relevance
        if all(field in sample for field in ["image", "conversations"]):
            relevance_score = await self._validate_image_relevance(sample)
            metric_scores["image_relevance"] = relevance_score

        # Calculate weighted overall score
        weights = {
            "conversation_structure": config.get("conversation_structure_weight", 1.0),
            "conversation_quality": config.get("conversation_quality_weight", 1.5),
            "image_relevance": config.get("image_relevance_weight", 1.0),
        }

        overall_score = self._calculate_weighted_score(metric_scores, weights)

        # Check conversation length constraints
        conversations = sample.get("conversations", [])
        min_conversations = config.get("min_conversations", 2)
        max_conversations = config.get("max_conversations", 8)

        if len(conversations) < min_conversations:
            issues.append(f"Too few conversations: {len(conversations)} < {min_conversations}")
        elif len(conversations) > max_conversations:
            warnings.append(f"Many conversations: {len(conversations)} > {max_conversations}")

        # Check for proper speaker alternation
        if len(conversations) >= 2:
            speaker_pattern = [conv.get("from", "") for conv in conversations]
            if not self._validate_speaker_alternation(speaker_pattern):
                issues.append("Invalid speaker alternation pattern")

        # Create metadata
        metadata = {
            "num_conversations": len(conversations),
            "total_conversation_length": sum(len(conv.get("value", "")) for conv in conversations),
            "has_image": bool(sample.get("image")),
            "qa_type": sample.get("type", ""),
            "speaker_pattern": [conv.get("from", "") for conv in conversations],
        }

        return self._create_result(
            sample=sample,
            overall_score=overall_score,
            metric_scores=metric_scores,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metadata=metadata,
        )

    async def _validate_conversation_structure(self, sample: Dict[str, Any]) -> float:
        """Validate the structure of conversations"""
        conversations = sample.get("conversations", [])

        if not conversations:
            return 0.0

        score = 8.0  # Default score

        # Check each conversation has required fields
        for i, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                score -= 2.0
                continue

            if "from" not in conv or "value" not in conv:
                score -= 1.5
            elif conv["from"] not in ["human", "gpt"]:
                score -= 1.0

            if not conv.get("value", "").strip():
                score -= 1.0

        # Check minimum length
        if len(conversations) < 2:
            score -= 2.0

        return max(0.0, min(10.0, score))

    async def _validate_conversation_quality(self, sample: Dict[str, Any]) -> float:
        """Validate the quality of conversations"""
        conversations = sample.get("conversations", [])

        if not conversations:
            return 0.0

        score = 7.0  # Default score

        # Check for question-answer pattern
        if len(conversations) >= 2:
            first_conv = conversations[0]
            if first_conv.get("from") == "human" and "?" in first_conv.get("value", ""):
                score += 1.0

        # Check conversation length appropriateness
        total_length = sum(len(conv.get("value", "")) for conv in conversations)
        if 50 <= total_length <= 1000:  # Reasonable length range
            score += 0.5

        return max(0.0, min(10.0, score))

    async def _validate_image_relevance(self, sample: Dict[str, Any]) -> float:
        """Validate image relevance to conversations"""
        image_path = sample.get("image", "")
        conversations = sample.get("conversations", [])

        if not image_path or not conversations:
            return 0.0

        score = 8.0  # Default score

        # Check if image file exists
        import os

        if not os.path.exists(image_path):
            score -= 3.0

        # Check if conversations mention spectrum analysis
        all_text = " ".join(conv.get("value", "") for conv in conversations).lower()
        spectrum_terms = ["spectrum", "peak", "signal", "nmr", "ir", "mass", "chemical shift"]

        if any(term in all_text for term in spectrum_terms):
            score += 1.0

        return max(0.0, min(10.0, score))

    def _validate_speaker_alternation(self, speaker_pattern: List[str]) -> bool:
        """Validate that speakers alternate properly"""
        if len(speaker_pattern) < 2:
            return False

        # Check that we start with human and alternate
        for i, speaker in enumerate(speaker_pattern):
            expected_speaker = "human" if i % 2 == 0 else "gpt"
            if speaker != expected_speaker:
                return False

        return True
