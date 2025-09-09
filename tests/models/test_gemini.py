"""Gemini Model Tests"""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env file is loaded
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

from src.models.gemini_api import Gemini


class TestGemini:
    def test_config_loading(self):
        """Test configuration loading"""
        from src.config import ModelConfig

        config = ModelConfig()

        # Check if config values exist (from environment variables or defaults)
        print(f"API Key status: {'Set' if config.gemini_api_key else 'Not Set'}")
        print(f"Base URL: {config.gemini_base_url}")
        print(f"Model Name: {config.gemini_model_name}")

        # At least default values should exist
        assert config.gemini_base_url is not None
        assert config.gemini_model_name is not None

    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires GEMINI_API_KEY environment variable")
    def test_model_real_api_call(self):
        """Test real API call"""
        # Validate environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        assert api_key is not None, "GEMINI_API_KEY environment variable not set"

        # Create model instance
        model = Gemini()

        # Print actual model information used
        print(f"Model used: {model.model_name}")
        print(f"API Endpoint: {model.base_url}")

        # Test simple text generation
        response = model.generate("Hello, please reply briefly", max_out_len=50)

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Model response: {response}")

    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires GEMINI_API_KEY environment variable")
    def test_multimodal_input(self):
        """Test multimodal input (requires a valid image file)"""
        # This test requires an actual image file
        test_image_path = project_root / "output" / "spectrum_images"

        # Find any available test image
        if test_image_path.exists():
            image_files = list(test_image_path.glob("*.png"))
            if image_files:
                test_image = str(image_files[0])

                model = Gemini()

                # Test multimodal input
                multimodal_prompt = {"text": "What type of spectrum is shown in this image?", "images": [test_image]}

                response = model.generate(multimodal_prompt, max_out_len=100)

                # Validate response
                assert isinstance(response, str)
                assert len(response) > 0
                print(f"Multimodal response: {response}")
            else:
                pytest.skip("No test images found")
        else:
            pytest.skip("Test image directory not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
