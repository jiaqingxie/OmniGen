from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from ..data_loaders.data_structures import DataSample


class BaseGenerator(ABC):
    """Base data generator class"""

    def __init__(self, config: Dict[str, Any], model_client=None):
        self.config = config
        self.model_client = model_client

    @abstractmethod
    def generate_single(self, sample: DataSample) -> Optional[Dict[str, Any]]:
        """Generate a single data sample

        Args:
            sample: Input data sample

        Returns:
            Generated data dictionary, or None if generation fails
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """Return the schema description of output data

        Returns:
            Dictionary describing the output format
        """
        pass

    def validate_input(self, sample: DataSample) -> bool:
        """Validate if input data meets requirements

        Args:
            sample: Input data sample

        Returns:
            Whether validation passes
        """
        return True

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate if output data meets requirements

        Args:
            output: Generated output data

        Returns:
            Whether validation passes
        """
        return output is not None and isinstance(output, dict)


class PromptTemplate:
    """Base prompt template class"""

    def __init__(self, template: str):
        self.template = template

    def build_prompt(self, sample: DataSample) -> str:
        """Build prompt

        Args:
            sample: Input data sample

        Returns:
            Built prompt
        """
        return self.template.format(**sample.metadata)


# Generator registry mechanism
_GENERATOR_REGISTRY: Dict[str, Type[BaseGenerator]] = {}


def register_generator(name: str):
    """Generator registration decorator

    Args:
        name: Generator name
    """

    def decorator(generator_class: Type[BaseGenerator]):
        if not issubclass(generator_class, BaseGenerator):
            raise ValueError(f"Generator {generator_class} must inherit from BaseGenerator")

        _GENERATOR_REGISTRY[name] = generator_class
        return generator_class

    return decorator


def get_generator(name: str) -> Optional[Type[BaseGenerator]]:
    """Get registered generator class

    Args:
        name: Generator name

    Returns:
        Generator class, or None if not found
    """
    return _GENERATOR_REGISTRY.get(name)


def list_generators() -> Dict[str, Type[BaseGenerator]]:
    """List all registered generators

    Returns:
        Mapping from generator names to classes
    """
    return _GENERATOR_REGISTRY.copy()


def create_generator(name: str, config: Dict[str, Any], model_client=None) -> Optional[BaseGenerator]:
    """Create generator instance

    Args:
        name: Generator name
        config: Configuration parameters
        model_client: Model client

    Returns:
        Generator instance, or None if creation fails
    """
    generator_class = get_generator(name)
    if generator_class is None:
        return None

    try:
        return generator_class(config, model_client)
    except Exception as e:
        print(f"Failed to create generator {name}: {e}")
        return None
