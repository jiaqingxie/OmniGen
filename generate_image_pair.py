#!/usr/bin/env python3
"""
Simple script to test image-pair generation.
This is a temporary script for testing - will be replaced by unified main module later.
"""

import asyncio
import argparse
from pathlib import Path
from src.config import OmniGenConfig
from src.core import OmniGenEngine


def main():
    parser = argparse.ArgumentParser(description="OmniGen Image-Pair Generator (Test)")
    parser.add_argument(
        "--config", type=str, default="src/config/image_pair.yaml", help="Path to image-pair config YAML"
    )
    parser.add_argument("--data-source", type=str, help="Override data source")
    parser.add_argument("--samples", type=int, help="Override number of samples")
    parser.add_argument("--output", type=str, help="Override output file path")
    parser.add_argument("--max-data-samples", type=int, help="Limit data samples loaded")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Load configuration
    try:
        config = OmniGenConfig.from_file(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Override settings from command line
    if args.data_source:
        # Use HuggingFace dataset
        config.dataset_path = args.data_source
        print(f"Using data source: {args.data_source}")

    if args.max_data_samples:
        if "loader_kwargs" not in config.generator_config:
            config.generator_config["loader_kwargs"] = {}
        config.generator_config["loader_kwargs"]["max_samples"] = args.max_data_samples
        print(f"Limiting data samples to: {args.max_data_samples}")

    if args.samples:
        config.num_samples = args.samples

    if args.output:
        config.output_path = args.output

    if args.verbose:
        config.verbose = True

    output_file = config.output_path
    print(f"Using output file: {output_file}")

    # Print configuration summary
    print("Configuration:")
    print(f"  - Data source: {config.dataset_path}")
    print(f"  - Generator type: {config.generator_type}")
    print(f"  - Target samples: {config.num_samples}")
    print(f"  - Output path: {config.output_path}")

    # Run generation
    async def run_generation():
        try:
            engine = OmniGenEngine(config)
            results = await engine.run()
            return results
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    results = asyncio.run(run_generation())

    if results:
        print(f"\n✅ Generation finished. {len(results)} samples generated.")
        print(f"Results saved to: {config.output_path}")

        # Show sample output
        if results and config.verbose:
            print("\nSample output:")
            sample = results[0]
            print(f"  ID: {sample.get('id', 'N/A')}")
            print(f"  Type: {sample.get('type', 'N/A')}")
            print(f"  Image: {sample.get('image', 'N/A')}")
            print(f"  Text: {sample.get('text', 'N/A')[:100]}...")
    else:
        print("❌ Generation failed.")


if __name__ == "__main__":
    main()
