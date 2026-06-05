"""Abstract base class for OCR backends."""

import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class TransientError(Exception):
    """Wraps a transient error that should be retried."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class Backend(ABC):
    """Abstract base class defining the backend interface for OCR processing."""

    # GLM-OCR task prompts
    PROMPTS = {
        "text": "Text Recognition:",
        "formula": "Formula Recognition:",
        "table": "Table Recognition:",
        "figure": "Figure Recognition:",
        # Aliases for consistency with other OCR CLIs
        "convert": "Text Recognition:",
        "ocr": "Text Recognition:",
    }

    def __init__(
        self,
        model_name: str,
        max_dimension: int | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.max_dimension = max_dimension
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model: bool = False

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier (e.g., 'ollama', 'vllm')."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Initialize connection to the backend and verify model availability."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Clean up backend connection."""
        ...

    @abstractmethod
    def process_image(
        self,
        image: Image.Image | Path | str,
        prompt: str | None = None,
        task: str = "text",
        return_raw: bool = False,
    ) -> str:
        """Process a single image and return OCR text.

        Args:
            image: PIL Image, file path, or path string
            prompt: Custom prompt (overrides task prompt)
            task: Task type (text, formula, table, figure)
            return_raw: If True, return raw model output without cleaning

        Returns:
            OCR text result
        """
        ...

    def _retry(self, func, *args, **kwargs):
        """Execute func with exponential backoff retry on TransientError."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except TransientError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning(
                        f"[{self.backend_name}] Transient error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[{self.backend_name}] Max retries ({self.max_retries}) exhausted: {e}"
                    )
        raise RuntimeError(
            f"Max retries ({self.max_retries}) exhausted: {last_error}"
        )

    def process_images_batch(
        self,
        images: list,
        prompt: str | None = None,
        task: str = "text",
    ) -> list:
        """Process multiple images sequentially.

        Default implementation; backends can override for optimization.
        """
        results = []
        for image in images:
            result = self.process_image(image, prompt=prompt, task=task)
            results.append(result)
        return results

    def describe_figure(self, image: Image.Image | Path | str) -> str:
        """Generate a description of a figure/chart/diagram."""
        return self.process_image(
            image,
            prompt=self.PROMPTS["figure"],
            task="figure",
        )

    def get_prompt(self, task: str) -> str:
        """Get the default prompt for a task."""
        return self.PROMPTS.get(task, self.PROMPTS["text"])
