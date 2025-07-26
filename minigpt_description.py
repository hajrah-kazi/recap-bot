import torch
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
)
import logging
from typing import Optional
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Configure logging for this module
logger = logging.getLogger("MiniGPTDescriptor")

class MiniGPTDescriptor:
    def __init__(self, model_type: str = "blip", device: Optional[str] = None):
        """
        Initialize captioning model with BLIP (preferred, robust).
        
        Args:
            model_type (str): Only "blip" is supported in this revision.
            device (str): "cuda", "mps" (Apple Silicon), or "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "blip"  # Force BLIP; minigpt/local loading removed for stability

        logger.info("Initializing MiniGPTDescriptor with BLIP image captioning")

        try:
            self._load_blip()
            logger.info("Loaded BLIP model")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {str(e)}")
            raise

    def _load_blip(self):
        """Load BLIP model (Salesforce/blip-image-captioning-base)"""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)

    def describe_frame(self, image_path: str, max_length: int = 150, **kwargs) -> str:
        """
        Generate a caption for an image frame using BLIP.
        
        Args:
            image_path (str): Path to input image
            max_length (int): Maximum caption length
            **kwargs: Additional model-specific parameters

        Returns:
            str: Generated caption
        """
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs, 
                max_length=max_length,
                num_beams=3,
                do_sample=False,
                early_stopping=True,
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.error(f"Captioning failed for {image_path}: {str(e)}")
            return "Description unavailable"

# Ensure logging is configured in the main application
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
