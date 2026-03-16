"""GLM-OCR CLI - OCR processing via Ollama or vLLM."""

__version__ = "0.1.0"
__author__ = "Ruben Fernandez Fuertes"
__license__ = "MIT"

from glm_ocr.backends import Backend, OllamaBackend, VLLMBackend, create_backend
from glm_ocr.processor import OCRProcessor

__all__ = [
    "OCRProcessor",
    "Backend",
    "OllamaBackend",
    "VLLMBackend",
    "create_backend",
]
