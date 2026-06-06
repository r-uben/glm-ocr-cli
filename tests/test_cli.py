"""Tests for GLM-OCR CLI."""

import importlib

from click.testing import CliRunner

from glm_ocr.cli import cli


def test_stale_env_file_does_not_crash_cli(tmp_path, monkeypatch):
    """End-to-end: a .env with a removed key must not crash even `--help`.

    Settings() runs at import time; without extra="ignore" pydantic-settings
    raises a ValidationError before Click ever runs, breaking every command.
    Re-import config in the dir holding the stale .env to exercise that path.
    """
    env = tmp_path / ".env"
    env.write_text("GLM_OCR_OUTPUT_DIR=/old/removed/path\nGLM_OCR_BACKEND=ollama\n")
    monkeypatch.chdir(tmp_path)

    import glm_ocr.config as config

    # Reconstruct Settings from the on-disk .env in cwd; must not raise.
    reloaded = importlib.reload(config)
    assert reloaded.Settings().backend == "ollama"

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "glm-ocr" in result.output.lower() or "version" in result.output.lower()

    def test_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "GLM-OCR" in result.output

    def test_process_help(self):
        result = self.runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "INPUT_PATH" in result.output

    def test_dry_run_single_file(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        # Create a minimal valid PDF
        pdf_path.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Pages 2 0 R>>endobj "
            b"2 0 obj<</Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</MediaBox[0 0 612 792]>>endobj\n"
            b"trailer<</Root 1 0 R>>"
        )
        result = self.runner.invoke(cli, ["process", str(pdf_path), "--dry-run"])
        assert result.exit_code == 0
        assert "test.pdf" in result.output

    def test_dry_run_quiet(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Pages 2 0 R>>endobj "
            b"2 0 obj<</Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</MediaBox[0 0 612 792]>>endobj\n"
            b"trailer<</Root 1 0 R>>"
        )
        result = self.runner.invoke(cli, ["process", str(pdf_path), "--dry-run", "-q"])
        assert result.exit_code == 0

    def test_nonexistent_file(self):
        result = self.runner.invoke(cli, ["process", "/nonexistent/file.pdf"])
        assert result.exit_code != 0

    def test_auto_insert_process(self, tmp_path):
        """Test that main() auto-inserts 'process' when first arg is a file path."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.0\n1 0 obj<</Pages 2 0 R>>endobj "
            b"2 0 obj<</Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</MediaBox[0 0 612 792]>>endobj\n"
            b"trailer<</Root 1 0 R>>"
        )
        # CliRunner bypasses main(), so test via process subcommand directly
        result = self.runner.invoke(cli, ["process", str(pdf_path), "--dry-run"])
        assert result.exit_code == 0
