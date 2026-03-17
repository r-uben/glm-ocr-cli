"""Tests for GLM-OCR configuration."""

from glm_ocr.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.backend == "ollama"
        assert s.model_name == "glm-ocr"
        assert s.ollama_url == "http://localhost:11434"
        assert s.vllm_base_url == "http://localhost:8080/v1"
        assert s.max_dimension == 2048
        assert s.max_retries == 3

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("GLM_OCR_BACKEND", "vllm")
        monkeypatch.setenv("GLM_OCR_MODEL_NAME", "custom-model")
        s = Settings()
        assert s.backend == "vllm"
        assert s.model_name == "custom-model"
