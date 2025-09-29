import os
from typing import Dict, Any, List
from .base import BaseValidator
from .registry import register_validator
from ..config.config import ValidatorConfig


@register_validator("cot")
class CoTValidator(BaseValidator):
    """Validator for Chain-of-Thought reasoning data."""

    def __init__(self, model_client, validation_config: ValidatorConfig):
        super().__init__(model_client, validation_config)

    def get_supported_data_type(self) -> str:
        """Get the data type this validator supports."""
        return "cot"

    def get_required_fields(self) -> List[str]:
        """Get the required fields for CoT data."""
        return ["id", "type", "question", "solution"]

    async def validate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single CoT sample."""
        issues = []
        warnings = []
        suggestions = []
        metric_scores = {}

        config = self.validation_config.cot_config

        # Validate basic structure
        if "question" in sample:
            question_score = await self._validate_question_quality(sample)
            metric_scores["question_quality"] = question_score

        if "solution" in sample:
            solution_score = await self._validate_solution_quality(sample)
            metric_scores["solution_quality"] = solution_score

        # Validate reasoning trajectories (only if they exist and are not empty)
        reasoning_fields = ["claude_thinking_trajectories", "interns1_thinking_trajectories"]
        has_reasoning = any(sample.get(field, "").strip() for field in reasoning_fields)
        if has_reasoning:
            reasoning_score = await self._validate_reasoning_quality(sample)
            metric_scores["reasoning_quality"] = reasoning_score

        # Validate model attempts (only if they exist and are not empty)
        attempt_fields = ["claude_attempt", "interns1_attempt"]
        has_attempts = any(sample.get(field, "").strip() for field in attempt_fields)
        if has_attempts:
            attempt_score = await self._validate_attempt_quality(sample)
            metric_scores["attempt_quality"] = attempt_score

        # Validate image relevance for multimodal
        if sample.get("type") == "cot multimodal" and "image" in sample:
            image_score = await self._validate_image_relevance(sample)
            metric_scores["image_relevance"] = image_score

        # Calculate weighted overall score
        weights = {
            "question_quality": config.get("question_quality_weight", 1.0),
            "solution_quality": config.get("solution_quality_weight", 1.5),
            "reasoning_quality": config.get("reasoning_quality_weight", 1.0),
            "attempt_quality": config.get("attempt_quality_weight", 1.0),
            "image_relevance": config.get("image_relevance_weight", 1.0),
        }

        # Only include weights for metrics that were actually calculated
        filtered_weights = {k: v for k, v in weights.items() if k in metric_scores}
        overall_score = self._calculate_weighted_score(metric_scores, filtered_weights)

        # Check for required model outputs (only if reasoning fields are present and not empty)
        reasoning_fields = [
            "claude_thinking_trajectories",
            "interns1_thinking_trajectories",
            "claude_attempt",
            "interns1_attempt",
        ]
        has_any_reasoning = any(sample.get(field, "").strip() for field in reasoning_fields)

        if has_any_reasoning:
            has_claude = any(
                sample.get(field, "").strip() for field in ["claude_thinking_trajectories", "claude_attempt"]
            )
            has_interns1 = any(
                sample.get(field, "").strip() for field in ["interns1_thinking_trajectories", "interns1_attempt"]
            )

            if not has_claude and not has_interns1:
                issues.append("Missing model reasoning outputs (need at least one model)")

        # Check content length
        question_length = len(sample.get("question", ""))
        solution_length = len(sample.get("solution", ""))

        if question_length < 20:
            issues.append(f"Question too short: {question_length} characters")
        elif question_length < 50:
            warnings.append(f"Question quite short: {question_length} characters")

        if solution_length < 50:
            issues.append(f"Solution too short: {solution_length} characters")
        elif solution_length < 100:
            warnings.append(f"Solution quite short: {solution_length} characters")

        # Check for multimodal requirements
        if sample.get("type") == "cot multimodal":
            if "image" not in sample:
                issues.append("Missing image for multimodal CoT")
            elif not os.path.exists(sample.get("image", "")):
                issues.append("Image file does not exist")

        # Determine validity
        is_valid = len(issues) == 0 and overall_score >= config.get("min_score", 5.0)

        # Create metadata
        metadata = {
            "question_length": question_length,
            "solution_length": solution_length,
            "has_claude": any(field in sample for field in ["claude_thinking_trajectories", "claude_attempt"]),
            "has_interns1": any(field in sample for field in ["interns1_thinking_trajectories", "interns1_attempt"]),
            "cot_type": sample.get("type", ""),
            "has_image": bool(sample.get("image")),
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

    async def _validate_question_quality(self, sample: Dict[str, Any]) -> float:
        """Validate the quality of the question."""
        question = sample.get("question", "")

        if not question:
            return 0.0

        score = 7.0  # Default score

        # Check for question indicators
        if "?" in question:
            score += 1.0

        # Check for scientific terminology
        scientific_terms = ["spectrum", "molecule", "structure", "analysis", "peak", "signal", "chemical"]
        term_count = sum(1 for term in scientific_terms if term.lower() in question.lower())

        if term_count >= 3:
            score += 1.0
        elif term_count >= 1:
            score += 0.5

        # Check question length appropriateness
        if 50 <= len(question) <= 500:
            score += 0.5

        return max(0.0, min(10.0, score))

    async def _validate_solution_quality(self, sample: Dict[str, Any]) -> float:
        """Validate the quality of the solution."""
        solution = sample.get("solution", "")

        if not solution:
            return 0.0

        score = 7.0  # Default score

        # Check for step-by-step structure
        step_indicators = ["step", "first", "then", "next", "finally", "therefore", "thus"]
        step_count = sum(1 for indicator in step_indicators if indicator.lower() in solution.lower())

        if step_count >= 3:
            score += 1.0
        elif step_count >= 1:
            score += 0.5

        # Check for scientific reasoning
        reasoning_terms = ["because", "due to", "indicates", "suggests", "confirms", "evidence"]
        reasoning_count = sum(1 for term in reasoning_terms if term.lower() in solution.lower())

        if reasoning_count >= 2:
            score += 1.0
        elif reasoning_count >= 1:
            score += 0.5

        # Check solution length appropriateness
        if 100 <= len(solution) <= 2000:
            score += 0.5

        return max(0.0, min(10.0, score))

    async def _validate_reasoning_quality(self, sample: Dict[str, Any]) -> float:
        """Validate the quality of reasoning trajectories."""
        trajectories = []

        if "claude_thinking_trajectories" in sample:
            trajectories.append(sample["claude_thinking_trajectories"])
        if "interns1_thinking_trajectories" in sample:
            trajectories.append(sample["interns1_thinking_trajectories"])

        if not trajectories:
            return 0.0

        total_score = 0.0
        for trajectory in trajectories:
            if not trajectory:
                continue

            score = 6.0  # Default score for each trajectory

            # Check for reasoning indicators
            reasoning_indicators = ["think", "consider", "analyze", "reason", "logic", "because", "therefore"]
            reasoning_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in trajectory.lower())

            if reasoning_count >= 3:
                score += 1.0
            elif reasoning_count >= 1:
                score += 0.5

            # Check trajectory length
            if 50 <= len(trajectory) <= 1000:
                score += 0.5

            total_score += score

        return max(0.0, min(10.0, total_score / len(trajectories)))

    async def _validate_attempt_quality(self, sample: Dict[str, Any]) -> float:
        """Validate the quality of model attempts."""
        attempts = []

        if "claude_attempt" in sample:
            attempts.append(sample["claude_attempt"])
        if "interns1_attempt" in sample:
            attempts.append(sample["interns1_attempt"])

        if not attempts:
            return 0.0

        total_score = 0.0
        for attempt in attempts:
            if not attempt:
                continue

            score = 6.0  # Default score for each attempt

            # Check for solution structure
            solution_indicators = ["answer", "solution", "result", "conclusion", "therefore"]
            solution_count = sum(1 for indicator in solution_indicators if indicator.lower() in attempt.lower())

            if solution_count >= 2:
                score += 1.0
            elif solution_count >= 1:
                score += 0.5

            # Check attempt length
            if 30 <= len(attempt) <= 800:
                score += 0.5

            total_score += score

        return max(0.0, min(10.0, total_score / len(attempts)))

    async def _validate_image_relevance(self, sample: Dict[str, Any]) -> float:
        """Validate image relevance for multimodal CoT."""
        image_path = sample.get("image", "")
        question = sample.get("question", "")
        solution = sample.get("solution", "")

        if not image_path or not question:
            return 0.0

        score = 8.0  # Default score

        # Check if image file exists
        if not os.path.exists(image_path):
            score -= 3.0

        # Check if question mentions spectrum analysis
        spectrum_terms = ["spectrum", "peak", "signal", "absorption", "frequency", "wavelength"]
        question_spectrum_count = sum(1 for term in spectrum_terms if term.lower() in question.lower())

        if question_spectrum_count >= 2:
            score += 1.0
        elif question_spectrum_count >= 1:
            score += 0.5

        # Check if solution mentions image analysis
        solution_image_count = sum(1 for term in spectrum_terms if term.lower() in solution.lower())

        if solution_image_count >= 1:
            score += 0.5

        return max(0.0, min(10.0, score))

    def _calculate_weighted_score(self, metric_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        if not metric_scores:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        for metric, score in metric_scores.items():
            weight = weights.get(metric, 1.0)
            total_weighted_score += score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0
