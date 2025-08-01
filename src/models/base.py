from abc import ABC, abstractmethod
from typing import Union, Any, Dict, List
from PIL import Image as PILImage


class BaseModel(ABC):
    """Base class for all models"""

    is_api: bool = False

    def __init__(self, path: str, max_seq_len: int = 2048):
        self.path = path
        self.max_seq_len = max_seq_len

    @abstractmethod
    def generate(self, prompt: Union[str, Dict[str, Any]], max_out_len: int = 512) -> str:
        """
        Generate response for a single prompt.

        Args:
            prompt: Input prompt, can be:
                   - str: Simple text prompt
                   - Dict: Multimodal prompt with format:
                     {
                         "text": "question text",
                         "images": [
                             {"type": "image_url", "image_url": {"url": "data:..."}},
                             # OR for PIL images:
                             {"type": "pil_image", "image": <PIL.Image object>}
                         ]
                     }
            max_out_len: Maximum output length

        Returns:
            Generated response string
        """
        pass

    def format_multimodal_prompt(self, text: str, images: Dict[str, Union[str, PILImage.Image]]) -> Dict[str, Any]:
        """
        Format a multimodal prompt from text and images.

        Args:
            text: Text content
            images: Dictionary of image name to image (file path or PIL Image)

        Returns:
            Formatted prompt dictionary
        """
        formatted_images = []

        for img_name, img_data in images.items():
            if isinstance(img_data, PILImage.Image):
                # Handle PIL Image objects
                formatted_images.append({"type": "pil_image", "image": img_data, "name": img_name})
            elif isinstance(img_data, str):
                # Handle file paths - convert to data URL
                import base64

                with open(img_data, 'rb') as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode()
                # Determine MIME type
                ext = img_data.lower().split('.')[-1]
                mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif'] else "image/png"

                formatted_images.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}, "name": img_name}
                )

        return {"text": text, "images": formatted_images}
