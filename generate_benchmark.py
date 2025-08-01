import asyncio
import argparse
from pathlib import Path
from src.config import OmniGenConfig
from src.core.engine import OmniGenEngine


def main():
    parser = argparse.ArgumentParser(description="OmniGen Benchmark Generator")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config YAML/JSON")
    parser.add_argument("--question-type", type=str, default=None, help="Question type to generate (e.g. type_cls)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate (override config)")
    parser.add_argument("--data-source", type=str, default=None, help="Override data source (Hugging Face dataset ID)")
    parser.add_argument("--max-data-samples", type=int, default=None, help="Maximum number of data samples to load")

    args = parser.parse_args()

    # Load configuration
    config = OmniGenConfig.from_file(args.config)

    # Override configuration with command line arguments
    if args.question_type:
        config.generator_config["question_types"] = [args.question_type]

    if args.samples is not None:
        config.num_samples = args.samples

    # Handle data source override
    if args.data_source is not None:
        config.dataset_path = args.data_source
        print(f"Using data source: {args.data_source}")

    # Handle max data samples
    if args.max_data_samples is not None:
        # Add loader_kwargs to generator_config if not exists
        if "loader_kwargs" not in config.generator_config:
            config.generator_config["loader_kwargs"] = {}
        config.generator_config["loader_kwargs"]["max_samples"] = args.max_data_samples
        print(f"Limiting data samples to: {args.max_data_samples}")

    # Print configuration summary
    print(f"Configuration:")
    print(f"  - Data source: {config.dataset_path}")
    print(f"  - Question types: {config.generator_config.get('question_types', [])}")
    print(f"  - Target samples: {config.num_samples}")
    print(f"  - Output path: {config.output_path}")

    # Create and run engine
    try:
        engine = OmniGenEngine(config)
        results = asyncio.run(engine.run())
        print(f"\n✅ Generation finished. {len(results)} samples generated.")
        print(f"Results saved to: {config.output_path}")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
