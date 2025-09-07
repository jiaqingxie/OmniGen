"""
Example script for validating generated data using the OmniGen validation system.

Usage:
    python validate_benchmark.py --config src/config/benchmark.yaml --data-type benchmark
    python validate_benchmark.py --file output/benchmark.json --data-type benchmark
    python validate_benchmark.py --file output/cot_data.json --data-type cot
"""

import asyncio
import argparse
from pathlib import Path
from src.config import OmniGenConfig
from src.core import ValidatorEngine
from src.validators import ValidatorResult


async def main():
    parser = argparse.ArgumentParser(description="Validate generated data")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--file", type=str, help="Path to data file to validate")
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        choices=["benchmark", "cot", "image_pair"],
        help="Type of data to validate",
    )
    parser.add_argument("--output-dir", type=str, help="Override output directory for validation reports")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = OmniGenConfig.from_file(args.config)
    else:
        # Use default config with validation enabled
        config = OmniGenConfig()
        config.validation_config.enabled = True

    # Override output directory if specified
    if args.output_dir:
        config.validation_config.output_dir = args.output_dir

    # Determine file to validate
    if args.file:
        data_file = args.file
    else:
        # Use config's output path
        data_file = config.output_path
        if not data_file.endswith('.json'):
            data_file += '.json'

    # Check if file exists
    if not Path(data_file).exists():
        print(f"Error: Data file not found: {data_file}")
        return

    print(f"Validating {args.data_type} data from: {data_file}")
    print(f"Using config: {args.config or 'default'}")

    # Create validation engine
    try:
        validation_engine = ValidatorEngine(config)

        # Run validation
        result = await validation_engine.validate_file(data_file, args.data_type)

        if not result.get("enabled", True):
            print("Validation was disabled in config")
            return

        # Print summary
        stats = result["statistics"]
        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Data type: {args.data_type}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid samples: {stats['valid_samples']} ({stats['validity_rate']:.1%})")
        print(f"Average score: {stats['average_score']:.2f}")

        print(f"\nSeverity distribution:")
        for level, count in stats['severity_distribution'].items():
            print(f"  {level.capitalize()}: {count}")

        if stats['common_issues']:
            print(f"\nTop issues:")
            for issue, count in list(stats['common_issues'].items())[:3]:
                print(f"  - {issue}: {count} samples")

        print(f"\nReports saved to: {config.validation_config.output_dir}")

    except Exception as e:
        print(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
