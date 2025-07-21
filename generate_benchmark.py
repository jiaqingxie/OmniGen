import asyncio
import argparse
from pathlib import Path
from src.config import OmniGenConfig
from src.core.engine import OmniGenEngine


def main():
    parser = argparse.ArgumentParser(description="OmniGen Benchmark Generator")
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config YAML/JSON")
    parser.add_argument("--question-type", type=str, required=True, help="Question type to generate (e.g. type_cls)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate (override config)")
    args = parser.parse_args()

    config = OmniGenConfig.from_file(args.config)
    # 覆盖 question_types，仅生成指定类型
    config.generator_config["question_types"] = [args.question_type]
    if args.samples is not None:
        config.num_samples = args.samples
    engine = OmniGenEngine(config)
    results = asyncio.run(engine.run())
    print(f"\n✅ Generation finished. {len(results)} samples generated.")
    print(f"Results saved to: {config.output_path}")


if __name__ == "__main__":
    main()
