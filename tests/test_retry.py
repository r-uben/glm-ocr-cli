"""Tests for retry logic."""

from unittest.mock import MagicMock, patch

import pytest

from glm_ocr.backends.base import Backend, TransientError


class ConcreteBackend(Backend):
    """Concrete backend for testing."""

    @property
    def backend_name(self) -> str:
        return "test"

    def load_model(self) -> None:
        self.model = True

    def unload_model(self) -> None:
        self.model = False

    def process_image(self, image, prompt=None, task="text", return_raw=False) -> str:
        return "test"


class TestRetry:
    def test_success_no_retry(self):
        backend = ConcreteBackend(model_name="test", max_retries=3, retry_delay=0.01)
        func = MagicMock(return_value="ok")
        result = backend._retry(func)
        assert result == "ok"
        assert func.call_count == 1

    @patch("time.sleep")
    def test_transient_then_success(self, mock_sleep):
        backend = ConcreteBackend(model_name="test", max_retries=3, retry_delay=0.01)
        func = MagicMock(side_effect=[TransientError("fail"), "ok"])
        result = backend._retry(func)
        assert result == "ok"
        assert func.call_count == 2

    @patch("time.sleep")
    def test_max_retries_exhausted(self, mock_sleep):
        backend = ConcreteBackend(model_name="test", max_retries=2, retry_delay=0.01)
        func = MagicMock(side_effect=TransientError("always fails"))
        with pytest.raises(RuntimeError, match="Max retries"):
            backend._retry(func)
        assert func.call_count == 3  # initial + 2 retries

    def test_non_transient_not_retried(self):
        backend = ConcreteBackend(model_name="test", max_retries=3, retry_delay=0.01)
        func = MagicMock(side_effect=ValueError("permanent"))
        with pytest.raises(ValueError):
            backend._retry(func)
        assert func.call_count == 1
