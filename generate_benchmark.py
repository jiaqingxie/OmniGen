import asyncio
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import OmniGenConfig
from src.core.engine import OmniGenEngine


def create_type_cls_config(num_samples: int = 10, model_type: str = "internvl") -> OmniGenConfig:
    """创建谱图类型识别任务的配置"""

    return OmniGenConfig(
        dataset_path="seed_datasets/molpuzzle/meta_data.json",
        generator_type="benchmark",
        generator_config={
            "question_types": ["type_cls"],
            "num_choices": 5,
            "prompt_templates": {
                "type_cls": """
请仔细观察这张谱图，识别它的类型。

这是一张分子谱图，请根据谱图的特征（如峰值位置、形状、强度等）来判断它属于哪种类型的谱图。

选项：
- Infrared Spectrum (IR) - 红外光谱，显示分子振动模式
- Proton Nuclear Magnetic Resonance (H-NMR) - 质子核磁共振，显示氢原子环境
- Carbon Nuclear Magnetic Resonance (C-NMR) - 碳核磁共振，显示碳原子环境
- Mass Spectrometry (MS) - 质谱，显示分子质量和碎片
- Raman Spectrum - 拉曼光谱，显示分子振动模式

请以JSON格式回复：
{{
    "question": "What type of spectrum is shown in this image?",
    "choices": ["Infrared Spectrum (IR)", "Proton Nuclear Magnetic Resonance (H-NMR)", "Carbon Nuclear Magnetic Resonance (C-NMR)", "Mass Spectrometry (MS)", "Raman Spectrum"],
    "answer": "正确答案"
}}
"""
            },
        },
        model_type=model_type,
        model_config={
            "api_key": os.getenv("INTERNVL_API_KEY"),
            "base_url": os.getenv("INTERNVL_BASE_URL"),
            "model_name": os.getenv("INTERNVL_MODEL_NAME"),
            "max_seq_len": 4096,
        },
        num_samples=num_samples,
        output_path="output/benchmark_type_cls.json",
        verbose=True,
        max_retries=3,
    )


def create_quality_eval_config(num_samples: int = 10, model_type: str = "internvl") -> OmniGenConfig:
    """创建谱图质量评估任务的配置"""

    return OmniGenConfig(
        dataset_path="seed_datasets/molpuzzle/meta_data.json",
        generator_type="benchmark",
        generator_config={
            "question_types": ["quality_eval"],
            "num_choices": 4,
            "prompt_templates": {
                "quality_eval": """
请评估这张谱图的信号质量。

仔细观察谱图的清晰度、噪声水平、峰值强度等特征，判断是否存在明显的质量问题。

选项：
- Yes, obvious quality issues - 是的，存在明显的质量问题
- No, the signal is very clear - 不，信号非常清晰
- Localized noise - 局部噪声
- Very low noise, negligible - 噪声很低，可忽略

请以JSON格式回复：
{{
    "question": "Does this spectrum show obvious signal quality issues?",
    "choices": ["Yes, obvious quality issues", "No, the signal is very clear", "Localized noise", "Very low noise, negligible"],
    "answer": "正确答案"
}}
"""
            },
        },
        model_type=model_type,
        model_config={
            "api_key": os.getenv("INTERNVL_API_KEY"),
            "base_url": os.getenv("INTERNVL_BASE_URL"),
            "model_name": os.getenv("INTERNVL_MODEL_NAME"),
            "max_seq_len": 4096,
        },
        num_samples=num_samples,
        output_path="output/benchmark_quality_eval.json",
        verbose=True,
        max_retries=3,
    )


async def generate_benchmark(benchmark_type: str, num_samples: int, model_type: str = "internvl"):
    """生成指定类型的 benchmark 数据"""

    print(f"🚀 开始生成 {benchmark_type} benchmark 数据...")
    print(f"📊 样本数量: {num_samples}")
    print(f"🤖 模型类型: {model_type}")

    if model_type == "internvl":
        api_key = os.getenv("INTERNVL_API_KEY")
        if not api_key or api_key == "your_actual_internvl_api_key_here":
            print("❌ 请先设置 INTERNVL_API_KEY 环境变量")
            print("1. 复制 env_template.txt 为 .env")
            print("2. 在 .env 文件中填写您的真实 API Key")
            return False

    # 根据类型创建配置
    if benchmark_type == "type_cls":
        config = create_type_cls_config(num_samples, model_type)
    elif benchmark_type == "quality_eval":
        config = create_quality_eval_config(num_samples, model_type)
    else:
        print(f"❌ 不支持的 benchmark 类型: {benchmark_type}")
        return False

    # 创建引擎
    engine = OmniGenEngine(config)

    try:
        results = await engine.run()

        print(f"✅ 生成完成，共生成 {len(results)} 个样本")

        if benchmark_type == "type_cls":
            print("\n📈 谱图类型分布:")
            type_counts = {}
            for result in results:
                answer = result.get('answer', 'Unknown')
                type_counts[answer] = type_counts.get(answer, 0) + 1

            for spectrum_type, count in type_counts.items():
                print(f"  {spectrum_type}: {count} 个")

        # 保存结果
        output_path = Path(config.output_path)
        print(f"💾 结果已保存到: {output_path}")

        return True

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OmniGen")
    parser.add_argument("--type", choices=["type_cls", "quality_eval"], default="type_cls")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--model", choices=["mock", "internvl"], default="internvl", help="使用的模型类型 (默认: internvl)"
    )

    args = parser.parse_args()
    Path("output").mkdir(exist_ok=True)
    success = asyncio.run(generate_benchmark(args.type, args.samples, args.model))
    if success:
        print("\n🎉 Benchmark 生成成功！")
    else:
        print("\n💥 Benchmark 生成失败！")


if __name__ == "__main__":
    main()
