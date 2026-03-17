"""Backend implementations for GLM-OCR."""

from glm_ocr.backends.base import Backend, TransientError
from glm_ocr.backends.ollama import OllamaBackend
from glm_ocr.backends.vllm import VLLMBackend

__all__ = [
    "Backend",
    "TransientError",
    "OllamaBackend",
    "VLLMBackend",
    "create_backend",
]


def create_backend(
    backend_type: str = "ollama",
    model_name: str = "glm-ocr",
    max_dimension: int | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    **kwargs,
) -> Backend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: "transformers" (default), "ollama", or "vllm"
        model_name: Model name/identifier
        max_dimension: Maximum image dimension for resizing
        max_retries: Maximum number of retries for transient errors
        retry_delay: Base delay in seconds between retries
        **kwargs: Additional backend-specific arguments

    Returns:
        Backend instance
    """
    backend_type = backend_type.lower()

    if backend_type == "transformers":
        from glm_ocr.backends.transformers import TransformersBackend

        # Map short names to HuggingFace model IDs
        hf_model = model_name if "/" in model_name else "zai-org/GLM-OCR"
        return TransformersBackend(
            model_name=hf_model,
            max_dimension=max_dimension,
            max_retries=max_retries,
            retry_delay=retry_delay,
            device=kwargs.get("device"),
        )
    elif backend_type == "ollama":
        return OllamaBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            max_retries=max_retries,
            retry_delay=retry_delay,
            ollama_url=kwargs.get("ollama_url"),
        )
    elif backend_type == "vllm":
        return VLLMBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            max_retries=max_retries,
            retry_delay=retry_delay,
            base_url=kwargs.get("vllm_base_url"),
        )
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Use 'transformers', 'ollama', or 'vllm'."
        )
