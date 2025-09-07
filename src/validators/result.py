from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ValidatorResult:
    """Validation result for a single sample"""

    sample_id: str
    data_type: str
    overall_score: float
    metric_scores: Dict[str, float]
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    is_valid: bool
    metadata: Dict[str, Any]

    @property
    def severity_level(self) -> str:
        """Get severity level based on overall score"""
        if self.overall_score >= 8.0:
            return "excellent"
        elif self.overall_score >= 7.0:
            return "good"
        elif self.overall_score >= 5.0:
            return "acceptable"
        else:
            return "poor"
