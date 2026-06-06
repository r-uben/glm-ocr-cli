"""Utility functions for GLM-OCR CLI."""

import logging
import re
from pathlib import Path

from ocr_output_contract import iter_input_files, resolve_output_root
from PIL import Image

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {PDF_EXTENSION}


def setup_logging(level: str = "WARNING", verbose: bool = False) -> logging.Logger:
    """Configure root logging.

    ``--verbose`` forces DEBUG; otherwise the ``level`` argument is honored (so
    ``GLM_OCR_LOG_LEVEL`` / ``settings.log_level`` actually takes effect — the
    audit found the level was previously discarded).
    """
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, str(level).upper(), logging.WARNING)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("glm_ocr")


def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_file(file_path: Path) -> bool:
    return file_path.suffix.lower() == PDF_EXTENSION


def collect_files(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    """Discover supported input files under ``input_path``, excluding outputs.

    Discovery is delegated to the contract's :func:`iter_input_files`, which
    recurses the tree and PRUNES the resolved output-root subtree so a re-run
    never re-ingests its own saved ``.md`` / figure / page-raster PNG outputs as
    fresh inputs (the HIGH self-ingestion bug). The output root is resolved here
    via the same :func:`resolve_output_root` the writer uses, so the exclusion
    targets the *real* output directory (``<input>/ocr/`` by default, or the
    ``-o`` override), not a path component that merely happens to be named
    ``ocr`` — important under the user's own ``.../toolkits/ocr/...`` tree.

    Discovery is always recursive for directory inputs (the old ``recursive``
    flag is dropped): batch trees are walked in full and the output subtree is
    the only thing excluded.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if input_path.is_file() and not is_supported_file(input_path):
        raise ValueError(
            f"Unsupported file type: {input_path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    output_root = resolve_output_root(input_path, output_dir)
    files = list(iter_input_files(input_path, output_root, suffixes=SUPPORTED_EXTENSIONS))

    if not files:
        raise ValueError(f"No supported files found in: {input_path}")

    return files


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def resize_image_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image if it exceeds max_dimension to prevent timeouts.

    Args:
        image: PIL Image to potentially resize
        max_dimension: Maximum allowed width or height. 0 disables resizing.

    Returns:
        Resized image if needed, otherwise the original image
    """
    if max_dimension <= 0:
        return image

    width, height = image.size
    if max(width, height) <= max_dimension:
        return image

    ratio = max_dimension / max(width, height)
    new_size = (int(width * ratio), int(height * ratio))

    logging.getLogger(__name__).info(
        f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]} "
        f"(max_dimension={max_dimension})"
    )

    return image.resize(new_size, Image.Resampling.LANCZOS)


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    filename = filename.strip(". ")

    if not filename:
        filename = "untitled"

    return filename


def ensure_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Inline HTML formatting tags the model occasionally emits as wrappers. Only
# these are stripped, so legitimate angle-bracket content the OCR must preserve
# (math inequalities like ``<x>``, XML/HTML code examples, generic ``<tag ...>``)
# survives. Matches an opening/closing tag for exactly one of these names.
_HTML_FORMAT_TAGS = (
    "b",
    "i",
    "u",
    "s",
    "em",
    "strong",
    "span",
    "div",
    "p",
    "br",
    "sub",
    "sup",
    "mark",
    "small",
    "font",
)
_HTML_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_HTML_FORMAT_TAGS) + r")(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)


def clean_ocr_output(text: str, raw: bool = False) -> str:
    """Clean GLM-OCR output: normalize whitespace and strip residual artifacts.

    GLM-OCR emits relatively clean markdown. We strip only a known set of inline
    HTML formatting tags (the audit flagged that blanket ``<...>`` stripping
    deleted legitimate math inequalities and XML/code). ``raw=True`` skips all
    cleaning and returns the model output verbatim.
    """
    if raw:
        return text

    # Strip only known inline HTML formatting tags, not arbitrary <...> spans.
    text = _HTML_TAG_RE.sub("", text)

    # Decode common HTML entities
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&nbsp;": " ",
        "&#39;": "'",
        "&#x27;": "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text
