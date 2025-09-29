"""
Unified generation entry point for OmniGen.
"""

import asyncio
import argparse
import sys
from pathlib import Path

from src.config import OmniGenConfig
from src.core import OmniGenEngine
from .common import (
    add_common_generation_args,
    add_type_specific_generation_args,
    load_config_for_type,
    apply_generation_overrides,
    print_config_summary,
    handle_error,
    validate_data_type,
)


async def run_generation(config: OmniGenConfig, data_type: str) -> bool:
    """Run the generation process."""
    try:
        # Create engine
        engine = OmniGenEngine(config)

        # Run generation
        print(f"\n🚀 Starting {data_type} generation...")
        results = await engine.run()

        if results:
            print(f"\n✅ Generation completed successfully!")
            print(f"   Generated {len(results)} samples")
            print(f"   Results saved to: {config.output_path}")

            # Show sample output if verbose
            if config.verbose and results:
                print(f"\n📋 Sample output:")
                sample = results[0]
                print(f"   ID: {sample.get('id', 'N/A')}")
                if data_type == "benchmark":
                    print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
                    print(f"   Answer: {sample.get('answer', 'N/A')}")
                elif data_type == "image_pair":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    print(f"   Image: {sample.get('image', 'N/A')}")
                    print(f"   Text: {sample.get('text', 'N/A')[:100]}...")
                elif data_type == "qa_pair":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    print(f"   Image: {sample.get('image', 'N/A')}")
                    conversations = sample.get('conversations', [])
                    print(f"   Conversations: {len(conversations)}")
                    if conversations:
                        print(f"   First Q: {conversations[0].get('value', 'N/A')[:100]}...")
                        if len(conversations) > 1:
                            print(f"   First A: {conversations[1].get('value', 'N/A')[:100]}...")
                elif data_type == "cot":
                    print(f"   Type: {sample.get('type', 'N/A')}")
                    if sample.get('type') == "cot multimodal":
                        print(f"   Image: {sample.get('image', 'N/A')}")
                    print(f"   Question: {sample.get('question', 'N/A')[:100]}...")
                    print(f"   Solution: {sample.get('solution', 'N/A')[:100]}...")
                    if 'claude_thinking_trajectories' in sample:
                        print(f"   Claude reasoning: {sample.get('claude_thinking_trajectories', 'N/A')[:100]}...")
                    if 'interns1_thinking_trajectories' in sample:
                        print(f"   InternS1 reasoning: {sample.get('interns1_thinking_trajectories', 'N/A')[:100]}...")
                else:
                    print(f"   Content: {str(sample)[:200]}...")

            return True
        else:
            print(f"\n❌ Generation failed - no valid samples generated")
            return False

    except Exception as e:
        handle_error(e, "generation")
        return False


def main():
    """Main entry point for unified generation."""
    parser = argparse.ArgumentParser(
        description="OmniGen Unified Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m main.generate --type benchmark --samples 10
  python -m main.generate --type image_pair --data-source "SpectrumWorld/molpuzzle-seed-datasets" --samples 5
  python -m main.generate --type benchmark --config custom_config.yaml --output my_benchmark.json
        """,
    )

    # Add common arguments
    add_common_generation_args(parser)

    # Parse arguments to get data type, then add type-specific arguments
    args, remaining = parser.parse_known_args()

    try:
        validate_data_type(args.type)

        # Add type-specific arguments and re-parse
        add_type_specific_generation_args(parser, args.type)
        args = parser.parse_args()

        # Load configuration
        config = load_config_for_type(args.type, args.config)

        # Apply overrides
        apply_generation_overrides(config, args, args.type)

        # Print configuration summary
        print_config_summary(config, args.type, "generation")

        # Run generation
        success = asyncio.run(run_generation(config, args.type))

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        handle_error(e, "argument parsing or configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()
