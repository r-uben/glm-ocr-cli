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

    def test_stale_env_var_does_not_crash(self, monkeypatch):
        """HIGH (migration regression): a removed field still present in a user's
        .env must NOT raise a pydantic ValidationError. Settings() runs at import
        time, so without extra="ignore" a stale GLM_OCR_OUTPUT_DIR (dropped in the
        canon migration) would crash EVERY command, including --help.
        """
        # Previously-documented, now-removed fields:
        monkeypatch.setenv("GLM_OCR_OUTPUT_DIR", "/some/old/path")
        monkeypatch.setenv("GLM_OCR_INCLUDE_METADATA", "false")
        monkeypatch.setenv("GLM_OCR_TOTALLY_UNKNOWN_KEY", "whatever")
        # Must construct cleanly (no ValidationError) and ignore the extras.
        s = Settings()
        assert s.backend == "ollama"
        assert not hasattr(s, "output_dir")
