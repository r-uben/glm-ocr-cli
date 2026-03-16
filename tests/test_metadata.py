"""Tests for GLM-OCR metadata tracking."""

import json

from glm_ocr.metadata import MetadataManager, _file_checksum


class TestFileChecksum:
    def test_consistent(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        c1 = _file_checksum(f)
        c2 = _file_checksum(f)
        assert c1 == c2
        assert c1.startswith("sha256:")

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _file_checksum(f1) != _file_checksum(f2)


class TestMetadataManager:
    def test_fresh_state(self, tmp_path):
        mm = MetadataManager(tmp_path)
        assert mm.files == {}

    def test_record_and_check(self, tmp_path):
        mm = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.0 test content")

        assert not mm.is_processed(f)

        mm.record(
            f,
            pages=5,
            processing_time=1.23,
            model="glm-ocr",
            backend="ollama",
            output_path="test/test.md",
        )

        assert mm.is_processed(f)

    def test_modified_file_not_processed(self, tmp_path):
        mm = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"original")

        mm.record(
            f,
            pages=1,
            processing_time=0.5,
            model="glm-ocr",
            backend="ollama",
            output_path="test/test.md",
        )

        # Modify the file
        f.write_bytes(b"modified")
        assert not mm.is_processed(f)

    def test_persistence(self, tmp_path):
        mm1 = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"content")

        mm1.record(
            f,
            pages=1,
            processing_time=0.5,
            model="glm-ocr",
            backend="ollama",
            output_path="test/test.md",
        )

        # Create new manager — should load from disk
        mm2 = MetadataManager(tmp_path)
        assert mm2.is_processed(f)

    def test_atomic_write(self, tmp_path):
        mm = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"content")

        mm.record(
            f,
            pages=1,
            processing_time=0.5,
            model="glm-ocr",
            backend="ollama",
            output_path="test/test.md",
        )

        # Verify JSON is valid
        data = json.loads((tmp_path / "metadata.json").read_text())
        assert data["version"] == "1"
        assert "test.pdf" in data["files"]
