"""Transformers backend for GLM-OCR (local HuggingFace inference)."""

import logging
from pathlib import Path
from typing import Union

from PIL import Image

from glm_ocr.backends.base import Backend, TransientError
from glm_ocr.config import settings
from glm_ocr.utils import clean_ocr_output, resize_image_if_needed

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "zai-org/GLM-OCR"


class TransformersBackend(Backend):
    """Local inference via HuggingFace Transformers on CPU/MPS/CUDA."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        max_dimension: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        device: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            max_dimension=max_dimension if max_dimension is not None else settings.max_dimension,
            max_retries=max_retries if max_retries is not None else settings.max_retries,
            retry_delay=retry_delay if retry_delay is not None else settings.retry_delay,
        )
        self._device = device
        self._model = None
        self._processor = None
        logger.info(f"Initialized TransformersBackend with model: {self.model_name}")

    @property
    def backend_name(self) -> str:
        return "transformers"

    def _resolve_device(self) -> str:
        """Auto-detect best available device."""
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load_model(self) -> None:
        """Load model and processor from HuggingFace."""
        if self.model:
            logger.info("Model already loaded")
            return

        try:
            import torch
            from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "transformers and torch are required for the transformers backend. "
                "Install with: pip install 'glm-ocr-cli[transformers]'"
            )

        device = self._resolve_device()
        logger.info(f"Loading {self.model_name} on {device}...")

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = GlmOcrForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model = True
        logger.info(f"Model loaded on {self._model.device}")

    def unload_model(self) -> None:
        """Free model memory."""
        self._model = None
        self._processor = None
        self.model = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """Run inference on a single image."""
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=8192)

        # Decode only the generated tokens (skip input)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def process_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str | None = None,
        task: str = "text",
        return_raw: bool = False,
    ) -> str:
        """Process image and return OCR text."""
        if not self.model or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")

        if isinstance(image, (Path, str)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

        if prompt is None:
            prompt = self.get_prompt(task)

        logger.debug(f"Processing image with prompt: {prompt}")

        image = resize_image_if_needed(image, self.max_dimension)

        raw_text = self._retry(self._generate, image, prompt)
        if return_raw:
            return raw_text
        return clean_ocr_output(raw_text)
