"""
OmniGen unified entry points module.

This module provides unified command-line interfaces for:
- Generation: python -m main.generate --type benchmark/image_pair/cot --config path/to/config.yaml
- Validation: python -m main.validate --type benchmark/image_pair/cot --file path/to/output.json

The unified entry points reduce code duplication and provide consistent
argument handling across different data types.
"""

__version__ = "1.0.0"
__all__ = ["common", "generate", "validate"]
