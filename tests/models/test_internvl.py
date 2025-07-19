"""InternVL 模型测试"""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# 确保加载 .env 文件
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

from src.models.internvl_api import InternVL


class TestInternVL:
    """InternVL 模型测试类"""

    def test_config_loading(self):
        """测试配置是否正确加载"""
        from src.config import ModelConfig

        config = ModelConfig()

        # 检查配置是否有值（从环境变量或默认值）
        print(f"API Key 状态: {'已设置' if config.internvl_api_key else '未设置'}")
        print(f"Base URL: {config.internvl_base_url}")
        print(f"Model Name: {config.internvl_model_name}")

        # 至少默认值应该存在
        assert config.internvl_base_url is not None
        assert config.internvl_model_name is not None

    @pytest.mark.skipif(not os.getenv("INTERNVL_API_KEY"), reason="需要 INTERNVL_API_KEY 环境变量")
    def test_model_real_api_call(self):
        """测试真实 API 调用"""
        # 验证环境变量
        api_key = os.getenv("INTERNVL_API_KEY")
        assert api_key is not None, "INTERNVL_API_KEY 环境变量未设置"

        # 创建模型实例
        model = InternVL()

        # 打印实际使用的模型信息
        print(f"使用的模型: {model.model_name}")
        print(f"API 端点: {model.base_url}")

        # 测试简单文本生成
        response = model.generate("你好，请简单回复", max_out_len=50)

        # 验证响应
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"模型响应: {response}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
