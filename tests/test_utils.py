"""Tests for GLM-OCR utility functions."""

import pytest
from PIL import Image

from glm_ocr.utils import (
    clean_ocr_output,
    collect_files,
    is_image_file,
    is_pdf_file,
    is_supported_file,
    resize_image_if_needed,
    sanitize_filename,
)


class TestFileDetection:
    def test_pdf(self, tmp_path):
        p = tmp_path / "test.pdf"
        p.touch()
        assert is_pdf_file(p)
        assert is_supported_file(p)
        assert not is_image_file(p)

    def test_image(self, tmp_path):
        for ext in (".jpg", ".png", ".webp"):
            p = tmp_path / f"test{ext}"
            p.touch()
            assert is_image_file(p)
            assert is_supported_file(p)
            assert not is_pdf_file(p)

    def test_unsupported(self, tmp_path):
        p = tmp_path / "test.txt"
        p.touch()
        assert not is_supported_file(p)


class TestCollectFiles:
    def test_single_file(self, tmp_path):
        p = tmp_path / "test.pdf"
        p.touch()
        files = collect_files(p)
        assert len(files) == 1

    def test_directory(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()
        files = collect_files(tmp_path)
        assert len(files) == 2

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.pdf").touch()
        (sub / "b.pdf").touch()
        files = collect_files(tmp_path, recursive=True)
        assert len(files) == 2

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            collect_files(tmp_path)


class TestSanitizeFilename:
    def test_clean(self):
        assert sanitize_filename("normal_file") == "normal_file"

    def test_special_chars(self):
        assert sanitize_filename("file<>:name") == "file___name"

    def test_empty(self):
        assert sanitize_filename("") == "untitled"


class TestResizeImage:
    def test_no_resize_needed(self):
        img = Image.new("RGB", (100, 100))
        result = resize_image_if_needed(img, 200)
        assert result.size == (100, 100)

    def test_resize(self):
        img = Image.new("RGB", (4000, 2000))
        result = resize_image_if_needed(img, 2000)
        assert max(result.size) <= 2000

    def test_disabled(self):
        img = Image.new("RGB", (4000, 2000))
        result = resize_image_if_needed(img, 0)
        assert result.size == (4000, 2000)


class TestCleanOutput:
    def test_html_entities(self):
        assert clean_ocr_output("&amp; &lt; &gt;") == "& < >"

    def test_html_format_tags_stripped(self):
        assert clean_ocr_output("<b>bold</b>") == "bold"
        assert clean_ocr_output("<em>x</em> <strong>y</strong>") == "x y"

    def test_math_inequalities_preserved(self):
        """SECONDARY fix: blanket <...> stripping killed math/XML; now preserved."""
        assert clean_ocr_output("if a < b then a <x> 0") == "if a < b then a <x> 0"

    def test_xml_and_code_preserved(self):
        assert clean_ocr_output("<config value='1'>") == "<config value='1'>"
        assert clean_ocr_output("<element>data</element>") == "<element>data</element>"

    def test_raw_optout_skips_cleaning(self):
        assert clean_ocr_output("<b>keep</b>\n\n\n\nx", raw=True) == "<b>keep</b>\n\n\n\nx"

    def test_excessive_newlines(self):
        assert clean_ocr_output("a\n\n\n\nb") == "a\n\nb"

    def test_strip(self):
        assert clean_ocr_output("  text  \n\n") == "text"
