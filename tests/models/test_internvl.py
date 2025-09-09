import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env file is loaded
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

from src.models.internvl_api import InternVL


class TestInternVL:
    def test_config_loading(self):
        from src.config import ModelConfig

        config = ModelConfig()

        # Check if config values exist (from environment variables or defaults)
        print(f"API Key status: {'Set' if config.internvl_api_key else 'Not Set'}")
        print(f"Base URL: {config.internvl_base_url}")
        print(f"Model Name: {config.internvl_model_name}")

        # At least default values should exist
        assert config.internvl_base_url is not None
        assert config.internvl_model_name is not None

    @pytest.mark.skipif(not os.getenv("INTERNVL_API_KEY"), reason="Requires INTERNVL_API_KEY environment variable")
    def test_model_real_api_call(self):
        """Test real API call"""
        # Validate environment variable
        api_key = os.getenv("INTERNVL_API_KEY")
        assert api_key is not None, "INTERNVL_API_KEY environment variable not set"

        # Create model instance
        model = InternVL()

        # Print actual model information used
        print(f"Model used: {model.model_name}")
        print(f"API Endpoint: {model.base_url}")

        # Test simple text generation
        response = model.generate("Hello, please reply briefly", max_out_len=50)

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Model response: {response}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
