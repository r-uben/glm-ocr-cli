"""Engine-level conformance tests: glm's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata writers,
the exit-code policy) are unit-tested inside the ``ocr-output-contract`` package
itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run glm's actual processor (with a
mocked backend so no GPU/Ollama is needed) over real inputs and assert the
produced output tree conforms to the family-wide contract via the package's
reusable :func:`ocr_output_contract.conformance.assert_conforms` harness, plus
the engine-specific fixes from the audit:

* NO YAML frontmatter on the markdown body (HIGH: the old to_markdown() prepended
  a ``---`` provenance block; the conformance harness rejects it).
* metadata keyed by input-RELATIVE path, not basename (HIGH: same-basename PDFs
  in different subdirs used to collide and clobber).
* nonzero exit on any failure/partial via RunOutcome (HIGH: batch used to exit 0
  even when every file failed).
* ``--task`` actually threads to the backend prompt (was a dead no-op).
"""

from __future__ import annotations

import json

import fitz
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms
from PIL import Image

from glm_ocr.backends.base import Backend
from glm_ocr.processor import process


class FakeBackend(Backend):
    """A mocked backend: deterministic per-page text, no network/model."""

    def __init__(self, text="OCR page text", model_name="glm-ocr"):
        super().__init__(model_name=model_name, max_retries=0, retry_delay=0.0)
        self.text = text
        self.model = True
        self.tasks_seen: list[str] = []
        self.prompts_seen: list[str | None] = []

    @property
    def backend_name(self) -> str:
        return "ollama"

    def load_model(self) -> None:
        self.model = True

    def unload_model(self) -> None:
        self.model = False

    def process_image(self, image, prompt=None, task="text", return_raw=False) -> str:
        self.tasks_seen.append(task)
        self.prompts_seen.append(prompt)
        # Mirror real backends: a None prompt resolves to the task prompt.
        return self.text


def _make_pdf(path, pages=2):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Source page {i + 1}")
    doc.save(path)
    doc.close()


def test_multipage_pdf_conforms_no_data_loss(tmp_path):
    """A 2-page PDF -> one <stem>/<stem>.md with both pages, no frontmatter."""
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    counter = {"n": 0}

    class PerPage(FakeBackend):
        def process_image(self, image, prompt=None, task="text", return_raw=False):
            counter["n"] += 1
            return f"PAGE-{counter['n']}-CONTENT"

    outcome = process(pdf, PerPage(), output_dir=out, show_progress=False)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )

    md = out / "sample" / "sample.md"
    body = md.read_text()
    assert "## Page 1" in body and "## Page 2" in body
    assert "PAGE-1-CONTENT" in body and "PAGE-2-CONTENT" in body
    # HIGH (audit): NO YAML frontmatter — body must not begin with a --- block.
    assert not body.lstrip().startswith("---")
    assert "source:" not in body and "processing_time:" not in body


def test_no_frontmatter_explicit(tmp_path):
    """Explicit no-frontmatter assertion: provenance lives only in the sidecar."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend(text="hello"), output_dir=out, show_progress=False)

    body = (out / "doc" / "doc.md").read_text()
    # The body is clean markdown that starts with the first page header.
    assert body.lstrip().startswith("## Page 1")
    # Provenance is in the JSON sidecar, not the markdown.
    sidecar = json.loads((out / "doc" / "metadata.json").read_text())
    assert sidecar["model"] == "glm-ocr"
    assert sidecar["backend"] == "ollama"
    assert sidecar["checksum"].startswith("sha256:")


def test_failed_document_recorded_and_exits_nonzero(tmp_path):
    """An empty model response -> status=failed, nonzero exit (no silent 0-byte ok)."""
    pdf = tmp_path / "blank.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome = process(pdf, FakeBackend(text="   "), output_dir=out, show_progress=False)
    assert outcome.exit_code != 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="blank.pdf", status="failed")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "blank" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"


def test_partial_pdf_conforms_with_partial_status(tmp_path):
    """One page ok, one empty -> partial, recorded, nonzero exit (uniform policy)."""
    pdf = tmp_path / "mixed.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    class Flaky(FakeBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def process_image(self, image, prompt=None, task="text", return_raw=False):
            self.calls += 1
            return "good content" if self.calls == 1 else ""

    outcome = process(pdf, Flaky(), output_dir=out, show_progress=False)
    assert outcome.exit_code != 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="mixed.pdf", pages=2, status="partial")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "mixed" / "metadata.json").read_text())
    assert "error" in doc_meta


def test_nested_batch_conforms_no_basename_collision(tmp_path):
    """Two same-basename PDFs in different subdirs both survive (input-relative key)."""
    root = tmp_path / "in"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    _make_pdf(root / "a" / "intro.pdf", pages=1)
    _make_pdf(root / "b" / "intro.pdf", pages=1)
    out = tmp_path / "out"

    outcome = process(root, FakeBackend(), output_dir=out, recursive=True, show_progress=False)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [
            ExpectedDoc(rel_key="a/intro.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="b/intro.pdf", pages=1, status="completed"),
        ],
    )
    # Root index keyed by input-relative path, not bare basename.
    index = json.loads((out / "metadata.json").read_text())
    assert set(index["files"]) == {"a/intro.pdf", "b/intro.pdf"}


def test_single_image_document_conforms(tmp_path):
    """A single image input is one conforming document keyed by its relative name."""
    img = tmp_path / "scan.png"
    Image.new("RGB", (60, 60), "white").save(img)
    out = tmp_path / "out"

    outcome = process(img, FakeBackend(), output_dir=out, show_progress=False)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="scan.png", pages=1, status="completed")],
    )


def test_default_output_root_is_input_parent_ocr(tmp_path):
    """No -o: output defaults to <input-parent>/ocr/ (never requires -o)."""
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf, pages=1)

    outcome = process(pdf, FakeBackend(), show_progress=False)
    assert outcome.exit_code == 0

    out = tmp_path / "ocr"
    assert (out / "paper" / "paper.md").exists()
    assert_conforms(out, [ExpectedDoc(rel_key="paper.pdf", pages=1, status="completed")])


def test_task_flag_threads_to_backend(tmp_path):
    """--task is no longer a dead no-op: it reaches the backend's process_image."""
    pdf = tmp_path / "f.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    backend = FakeBackend(text="formula text")
    process(pdf, backend, output_dir=out, task="formula", show_progress=False)
    assert backend.tasks_seen == ["formula"]
    # No custom prompt -> backend resolves the task prompt itself (prompt is None).
    assert backend.prompts_seen == [None]


def test_reprocess_idempotency(tmp_path):
    """A completed doc is skipped on re-run unless --reprocess is set."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    b1 = FakeBackend()
    process(pdf, b1, output_dir=out, show_progress=False)
    first_calls = len(b1.tasks_seen)
    assert first_calls == 1

    # Second run without --reprocess: skipped (no new backend calls).
    b2 = FakeBackend()
    out2 = process(pdf, b2, output_dir=out, show_progress=False)
    assert b2.tasks_seen == []
    assert out2.exit_code == 0

    # With --reprocess: re-OCR'd.
    b3 = FakeBackend()
    process(pdf, b3, output_dir=out, reprocess=True, show_progress=False)
    assert len(b3.tasks_seen) == 1
