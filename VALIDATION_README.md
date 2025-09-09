# OmniGen Validation System

The OmniGen validation system provides automated quality assessment for generated data including benchmark Q&A, Chain-of-Thought reasoning, and image pairs.

## Architecture Overview

```
src/
├── config/config.py          # ValidationConfig and integration with OmniGenConfig
├── core/validator_engine.py  # Main validation engine
└── validators/
    ├── __init__.py           # Module exports
    ├── base.py               # BaseValidator abstract class
    ├── registry.py           # Validator registration system
    ├── benchmark_validator.py     # Q&A benchmark validation
    ├── cot_validator.py          # Chain-of-Thought validation
    └── image_pair_validator.py   # Image pair validation
```

## Usage

### Basic Validation

```python
from src.config import OmniGenConfig
from src.core.validator_engine import ValidatorEngine

# Load config
config = OmniGenConfig.from_file("config.yaml")

# Create validation engine
validator_engine = ValidatorEngine(config)

# Validate benchmark data
result = await validator_engine.validate_file("output/benchmark.json", "benchmark")

# Check results
stats = result["statistics"]
print(f"Valid samples: {stats['valid_samples']}/{stats['total_samples']}")
print(f"Average score: {stats['average_score']:.2f}")
```

### Command Line Usage

```bash
# Validate benchmark data
python -m main.validate --type benchmark --file output/benchmark.json

# Validate image-text pair data
python -m main.validate --type image_pair --file output/image_pairs.json

# Validate with verbose output
python -m main.validate --type benchmark --file output/benchmark.json --verbose

# Validate with custom config
python -m main.validate --type benchmark --file output/benchmark.json --config custom_config.yaml
```

## Configuration

### Validation Settings

```yaml
validation_config:
  enabled: true
  output_dir: "validation_reports"
  verbose: true
  overall_threshold: 7.0
  critical_threshold: 5.0
  
  # Benchmark-specific settings
  benchmark_config:
    question_quality_weight: 1.5
    answer_correctness_weight: 2.0
    choices_quality_weight: 1.0
    image_relevance_weight: 1.2
    strict_mode: false
  
  # COT-specific settings
  cot_config:
    reasoning_coherence_weight: 2.0
    step_completeness_weight: 1.5
    factual_accuracy_weight: 2.0
    min_reasoning_steps: 3
    max_reasoning_steps: 10
  
  # Image pair settings
  image_pair_config:
    semantic_relevance_weight: 2.0
    quality_consistency_weight: 1.0
    min_resolution: 224
    supported_formats: ["png", "jpg", "jpeg"]
```

## Validation Metrics

### Benchmark Validation

- **Question Quality**: Clarity, grammar, and appropriateness
- **Answer Correctness**: Accuracy using LLM re-evaluation
- **Choices Quality**: Distractor appropriateness and variety
- **Image Relevance**: Alignment between images and questions

### COT Validation

- **Reasoning Coherence**: Logical flow between steps
- **Step Completeness**: Coverage of necessary reasoning
- **Factual Accuracy**: Correctness of facts and conclusions

### Image Pair Validation

- **Image Validity**: File existence and format compliance
- **Semantic Relevance**: Alignment between images and labels
- **Quality Consistency**: Similar quality across image pairs

## Output Reports

Validation generates:

- **JSON Report**: Detailed results with per-sample scores
- **Summary Text**: Human-readable overview
- **Statistics**: Aggregate metrics and common issues

Example output structure:

```
validation_reports/
├── benchmark_benchmark_validation.json
├── benchmark_benchmark_summary.txt
└── cot_cot_validation.json
```

## Extending the System

### Adding New Validators

1. Create a new validator class inheriting from `BaseValidator`
2. Register it using the `@register_validator` decorator
3. Implement required methods:
   - `get_supported_data_type()`
   - `get_required_fields()`
   - `validate_sample()`

```python
from src.validators.base import BaseValidator
from src.validators.registry import register_validator

@register_validator("my_data_type")
class MyValidator(BaseValidator):
    def get_supported_data_type(self) -> str:
        return "my_data_type"
    
    def get_required_fields(self) -> List[str]:
        return ["field1", "field2"]
    
    async def validate_sample(self, sample: Dict[str, Any]) -> ValidatorResult:
        # Implementation here
        pass
```

### Adding New Metrics

Add new validation dimensions by:

1. Extending the validator's `validate_sample()` method
2. Adding corresponding configuration in `ValidatorConfig`
3. Updating weights in the scoring calculation

## Integration with Generation Pipeline

The validation system integrates seamlessly with the existing OmniGen pipeline:

1. Generate data using `OmniGenEngine`
2. Validate output using `ValidatorEngine`
3. Review validation reports
4. Iterate on generation parameters if needed

This ensures high-quality data generation with automated quality control.
