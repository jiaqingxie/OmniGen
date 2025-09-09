from typing import Dict, Type, Optional, List
from .base import BaseValidator
from ..config.config import ValidatorConfig

# Global validator registry
_VALIDATOR_REGISTRY: Dict[str, Type[BaseValidator]] = {}


def register_validator(data_type: str):
    """Decorator to register a validator for a specific data type"""

    def decorator(validator_class: Type[BaseValidator]):
        if not issubclass(validator_class, BaseValidator):
            raise ValueError(f"Validator {validator_class} must inherit from BaseValidator")

        _VALIDATOR_REGISTRY[data_type] = validator_class
        return validator_class

    return decorator


def create_validator(data_type: str, model_client, validation_config: ValidatorConfig) -> Optional[BaseValidator]:
    """Create validator instance for the specified data type"""
    if data_type not in _VALIDATOR_REGISTRY:
        print(f"No validator registered for data type: {data_type}")
        return None

    validator_class = _VALIDATOR_REGISTRY[data_type]
    try:
        return validator_class(model_client, validation_config)
    except Exception as e:
        print(f"Failed to create validator for {data_type}: {e}")
        return None


def list_validators() -> Dict[str, Type[BaseValidator]]:
    """Get all registered validators"""
    return _VALIDATOR_REGISTRY.copy()


def get_supported_data_types() -> List[str]:
    """Get list of supported data types"""
    return list(_VALIDATOR_REGISTRY.keys())
