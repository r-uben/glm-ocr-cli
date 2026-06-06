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

import io
import json
from pathlib import Path

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

    outcome = process(root, FakeBackend(), output_dir=out, show_progress=False)
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
    # Quiet-mode scripting: the cached skip still emits the existing .md path.
    assert out2.outputs == [str(out / "doc" / "doc.md")]

    # With --reprocess: re-OCR'd.
    b3 = FakeBackend()
    process(pdf, b3, output_dir=out, reprocess=True, show_progress=False)
    assert len(b3.tasks_seen) == 1


def test_deleted_figure_forces_reprocess_not_skip(tmp_path):
    """HIGH: a cached skip re-validates only the markdown, so a deleted figure PNG
    leaves the body referencing a missing file (which the v0.1.3 conformance
    harness rejects) yet the doc would skip and exit 0.

    After a completed --analyze-figures run, deleting a referenced figure PNG
    must force a REPROCESS on the next run (not a skip), regenerating the figure
    so the body's inline link resolves again.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_figure(pdf)
    out = tmp_path / "out"

    b1 = FakeBackend()
    out1 = process(pdf, b1, output_dir=out, analyze_figures=True, show_progress=False)
    assert out1.exit_code == 0
    first_calls = len(b1.tasks_seen)
    assert first_calls >= 1

    # The completed run wrote a figure PNG the markdown links to.
    figures_dir = out / "doc" / "figures"
    fig_pngs = list(figures_dir.glob("*.png"))
    assert fig_pngs, "expected --analyze-figures to write a figure PNG"
    body = (out / "doc" / "doc.md").read_text()
    assert "figures/" in body, "the body should reference the figure inline"

    # Sanity: an unchanged re-run is skipped (no new backend calls).
    b_skip = FakeBackend()
    out_skip = process(pdf, b_skip, output_dir=out, analyze_figures=True, show_progress=False)
    assert b_skip.tasks_seen == [], "unchanged re-run must still be skipped"
    assert out_skip.exit_code == 0

    # Now DELETE the figure PNG -> the body has a dangling inline link.
    for png in fig_pngs:
        png.unlink()

    # Re-run: the cached skip must NOT fire (the side output is gone); the doc is
    # reprocessed, the backend is called again, and the figure is regenerated.
    b2 = FakeBackend()
    out2 = process(pdf, b2, output_dir=out, analyze_figures=True, show_progress=False)
    assert len(b2.tasks_seen) >= 1, "a deleted figure must force reprocess, not skip"
    assert list(figures_dir.glob("*.png")), "reprocess must regenerate the figure PNG"
    # And the regenerated output conforms (no dangling inline link remains).
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="doc.pdf", pages=1, status="completed")],
        require_failures_nonzero_exit=out2.exit_code != 0,
    )


def test_rerun_under_different_task_reprocesses(tmp_path):
    """Run-fingerprint idempotency: a re-run under a different --task does NOT
    silently reuse the prior output (it reprocesses)."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    b1 = FakeBackend()
    process(pdf, b1, output_dir=out, task="text", show_progress=False)
    assert b1.tasks_seen == ["text"]

    # Same input, different task -> fingerprint changes -> reprocessed.
    b2 = FakeBackend()
    process(pdf, b2, output_dir=out, task="formula", show_progress=False)
    assert b2.tasks_seen == ["formula"]

    # Re-running again under the SAME task is still skipped.
    b3 = FakeBackend()
    out3 = process(pdf, b3, output_dir=out, task="formula", show_progress=False)
    assert b3.tasks_seen == []
    assert out3.exit_code == 0


def test_rerun_under_different_model_reprocesses(tmp_path):
    """A re-run under a different model also invalidates the cache."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend(model_name="glm-ocr"), output_dir=out, show_progress=False)

    b2 = FakeBackend(model_name="glm-ocr-v2")
    process(pdf, b2, output_dir=out, show_progress=False)
    assert b2.tasks_seen == ["text"]  # reprocessed, not skipped


def test_fingerprint_recorded_in_metadata(tmp_path):
    """The run-config fingerprint is persisted so is_completed can compare it."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    process(pdf, FakeBackend(), output_dir=out, show_progress=False)
    sidecar = json.loads((out / "doc" / "metadata.json").read_text())
    assert sidecar.get("fingerprint", "").startswith("fp:")
    index = json.loads((out / "metadata.json").read_text())
    assert index["files"]["doc.pdf"]["fingerprint"] == sidecar["fingerprint"]


def test_rerun_does_not_reingest_output_dir(tmp_path):
    """HIGH (self-ingestion): a second recursive run with --extract-images /
    --analyze-figures must NOT re-discover its own saved PNG outputs as inputs.

    The default output root (<input>/ocr/) sits inside the scanned tree; without
    excluding it, the engine's page-raster / figure PNGs would be OCR'd again and
    pollute the tree recursively. Discovery via iter_input_files prunes the
    resolved output subtree, so the run-2 root index is unchanged.
    """
    root = tmp_path / "papers"
    root.mkdir()
    pdf = root / "paper.pdf"
    # A PDF with an embedded image so --analyze-figures writes a figure PNG.
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((40, 60), "Source page 1")
    img = Image.new("RGB", (40, 40), "blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    page.insert_image(fitz.Rect(100, 100, 180, 180), stream=buf.getvalue())
    doc.save(pdf)
    doc.close()

    # Run 1: default output root is <input>/ocr/ (inside the scanned dir).
    out1 = process(
        root,
        FakeBackend(),
        extract_images=True,
        analyze_figures=True,
        show_progress=False,
    )
    assert out1.exit_code == 0
    output_root = root / "ocr"
    # Sanity: prior-run PNG outputs now live INSIDE the scanned tree.
    pngs = list(output_root.rglob("*.png"))
    assert pngs, "expected --extract-images/--analyze-figures to write PNG outputs"

    index_after_run1 = json.loads((output_root / "metadata.json").read_text())
    keys_run1 = set(index_after_run1["files"])
    assert keys_run1 == {"paper.pdf"}

    # Run 2 over the SAME directory: must not ingest any of those PNGs.
    out2 = process(
        root,
        FakeBackend(),
        extract_images=True,
        analyze_figures=True,
        show_progress=False,
    )
    assert out2.exit_code == 0

    index_after_run2 = json.loads((output_root / "metadata.json").read_text())
    keys_run2 = set(index_after_run2["files"])
    assert keys_run2 == {"paper.pdf"}, (
        f"self-ingestion: run 2 discovered extra inputs {keys_run2 - keys_run1}"
    )
    # No nested ocr/ocr/ pollution.
    assert not (output_root / "ocr").exists()


def test_same_stem_different_ext_no_collision(tmp_path):
    """v0.1.1 doc_dir_for disambiguates a/foo.pdf vs a/foo.png so the second does
    not silently overwrite the first on disk."""
    root = tmp_path / "in"
    root.mkdir()
    _make_pdf(root / "foo.pdf", pages=1)
    Image.new("RGB", (40, 40), "white").save(root / "foo.png")
    out = tmp_path / "out"

    outcome = process(root, FakeBackend(), output_dir=out, show_progress=False)
    assert outcome.exit_code == 0

    # Distinct on-disk doc dirs: <out>/foo/ (pdf) and <out>/foo_png/ (image).
    assert (out / "foo" / "foo.md").exists()
    assert (out / "foo_png" / "foo.md").exists()

    assert_conforms(
        out,
        [
            ExpectedDoc(rel_key="foo.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="foo.png", pages=1, status="completed"),
        ],
    )


def test_raw_mode_skips_cleaning(tmp_path):
    """--raw plumbs return_raw=True end-to-end so verbatim model output survives."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    seen_raw: list[bool] = []

    class RawAware(FakeBackend):
        def process_image(self, image, prompt=None, task="text", return_raw=False):
            seen_raw.append(return_raw)
            # Mimic a real backend: cleaning would strip <b> tags; raw keeps them.
            return "<b>kept</b>" if return_raw else "stripped"

    process(pdf, RawAware(), output_dir=out, raw=True, show_progress=False)
    assert seen_raw == [True]
    body = (out / "doc" / "doc.md").read_text()
    assert "<b>kept</b>" in body


# ---------------------------------------------------------------------------
# HIGH (round-2): the idempotency fingerprint must include EVERY output-affecting
# flag, so a re-run that changes the output is reprocessed, never silently
# skipped. One test per flag: changing it must invalidate the cache (the backend
# is called again on the second run).
# ---------------------------------------------------------------------------


def _make_pdf_with_figure(path):
    """A 1-page PDF carrying an embedded image (so --analyze-figures has work)."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((40, 60), "Source page 1")
    img = Image.new("RGB", (40, 40), "blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    page.insert_image(fitz.Rect(100, 100, 180, 180), stream=buf.getvalue())
    doc.save(path)
    doc.close()


def _run_then_rerun(pdf, out, first_kwargs, second_kwargs):
    """Run twice (fresh backend each time) and return the 2nd run's call count.

    A nonzero count means the second invocation was NOT skipped, i.e. the flag
    change invalidated the fingerprint and the doc was reprocessed.
    """
    b1 = FakeBackend()
    for k, v in first_kwargs.pop("_backend_attrs", {}).items():
        setattr(b1, k, v)
    process(pdf, b1, output_dir=out, show_progress=False, **first_kwargs)

    b2 = FakeBackend()
    for k, v in second_kwargs.pop("_backend_attrs", {}).items():
        setattr(b2, k, v)
    process(pdf, b2, output_dir=out, show_progress=False, **second_kwargs)
    return len(b2.tasks_seen)


def test_rerun_under_different_raw_reprocesses(tmp_path):
    """--raw changes page text (verbatim vs cleaned); flipping it must reprocess."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    n = _run_then_rerun(pdf, out, {"raw": False}, {"raw": True})
    assert n == 1, "flipping --raw must invalidate the fingerprint and reprocess"


def test_rerun_under_different_dpi_reprocesses(tmp_path):
    """A different --dpi renders different pixels; the cache must not be reused."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    n = _run_then_rerun(pdf, out, {"dpi": 200}, {"dpi": 300})
    assert n == 1, "changing --dpi must invalidate the fingerprint and reprocess"


def test_rerun_under_different_max_dim_reprocesses(tmp_path):
    """--max-dim (carried on the backend) changes rendered pixels; must reprocess."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    n = _run_then_rerun(
        pdf,
        out,
        {"_backend_attrs": {"max_dimension": 2048}},
        {"_backend_attrs": {"max_dimension": 1024}},
    )
    assert n == 1, "changing --max-dim must invalidate the fingerprint and reprocess"


def test_rerun_under_different_analyze_figures_reprocesses(tmp_path):
    """--analyze-figures requests a NEW side output; enabling it must reprocess."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_figure(pdf)
    out = tmp_path / "out"

    # >=1 (not ==1): with --analyze-figures the second run makes an EXTRA
    # backend call for the figure caption on top of the page-OCR call, so the
    # count is >1. The point is it reprocessed at all (was not skipped).
    n = _run_then_rerun(pdf, out, {"analyze_figures": False}, {"analyze_figures": True})
    assert n >= 1, "enabling --analyze-figures must invalidate the fingerprint and reprocess"


def test_rerun_under_different_extract_images_reprocesses(tmp_path):
    """--extract-images requests page rasters a prior run never produced; reprocess."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    n = _run_then_rerun(pdf, out, {"extract_images": False}, {"extract_images": True})
    assert n == 1, "enabling --extract-images must invalidate the fingerprint and reprocess"


def test_same_flags_still_skips(tmp_path):
    """Control: identical output-affecting flags on a re-run are STILL skipped
    (the richer fingerprint did not break idempotency for unchanged configs)."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    n = _run_then_rerun(
        pdf,
        out,
        {"raw": True, "dpi": 150, "_backend_attrs": {"max_dimension": 1024}},
        {"raw": True, "dpi": 150, "_backend_attrs": {"max_dimension": 1024}},
    )
    assert n == 0, "identical flags must remain idempotent (skipped on re-run)"


# ---------------------------------------------------------------------------
# LOW (round-2): a figure-save failure must be VISIBLE — it degrades the doc to
# partial (not a silent completed) and drives a nonzero exit, with a non-empty
# diagnostic in the metadata.
# ---------------------------------------------------------------------------


def test_figure_save_failure_degrades_status_and_exits_nonzero(tmp_path, monkeypatch):
    """A failing figure save -> status=partial, nonzero exit, diagnostic recorded.

    Pages OCR cleanly, so without surfacing the figure-save failure the doc would
    record a silent ``completed``/exit 0 despite the user asking for figures.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_figure(pdf)
    out = tmp_path / "out"

    # Force PIL Image.save to raise so the figure cannot be written to disk.
    real_save = Image.Image.save

    def boom(self, *args, **kwargs):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(Image.Image, "save", boom)
    try:
        outcome = process(
            pdf, FakeBackend(), output_dir=out, analyze_figures=True, show_progress=False
        )
    finally:
        monkeypatch.setattr(Image.Image, "save", real_save)

    # The figure-save failure is VISIBLE: nonzero exit, partial status.
    assert outcome.exit_code != 0, "a figure-save failure must drive a nonzero exit"
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="doc.pdf", pages=1, status="partial")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "partial"
    assert doc_meta.get("error"), "the partial must carry a non-empty diagnostic"
    assert "save failed" in doc_meta["error"]


def test_figure_extraction_wholesale_crash_degrades_status_and_exits_nonzero(tmp_path, monkeypatch):
    """HIGH: a WHOLESALE figure-extraction crash under --analyze-figures must NOT
    be a silent completed/exit 0.

    The figure-extraction body crashes entirely (not a single bad image): it
    returns no figures, so the old ``if figures:`` guard never degraded the doc
    and the run recorded ``completed``/exit 0 — silently dropping the requested
    figure channel. The fix surfaces the crash as an extraction error folded into
    ``figure_errors`` unconditionally, so the doc degrades to ``partial`` with a
    diagnostic and the run exits nonzero.
    """
    import glm_ocr.processor as proc

    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_figure(pdf)
    out = tmp_path / "out"

    real_open = proc.fitz.open
    calls = {"n": 0}

    def flaky_open(*args, **kwargs):
        # 1st open = _pdf_to_images (page rendering): must succeed.
        # 2nd open = _extract_figures_from_pdf: crash the whole extraction body.
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("PDF figure extraction backend crashed (simulated)")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(proc.fitz, "open", flaky_open)

    outcome = process(pdf, FakeBackend(), output_dir=out, analyze_figures=True, show_progress=False)

    # Pages OCR cleanly but the entire figure channel was lost -> partial, NOT
    # a silent completed; nonzero exit; a non-empty diagnostic naming the crash.
    assert outcome.exit_code != 0, "a wholesale figure-extraction crash must drive a nonzero exit"
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="doc.pdf", pages=1, status="partial")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "partial"
    assert "figure extraction failed" in (doc_meta.get("error") or "")


def test_figure_analysis_failure_degrades_status_and_exits_nonzero(tmp_path):
    """A failing figure CAPTION (under --analyze-figures) is likewise visible:
    status=partial + nonzero exit, not a silent completed."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_figure(pdf)
    out = tmp_path / "out"

    class CaptionFails(FakeBackend):
        def process_image(self, image, prompt=None, task="text", return_raw=False):
            # Page OCR (task='text') succeeds; the figure caption call (the long
            # description prompt, task default) raises.
            if prompt and "Describe" in prompt:
                raise RuntimeError("caption model unavailable")
            return "page text"

    outcome = process(
        pdf, CaptionFails(), output_dir=out, analyze_figures=True, show_progress=False
    )
    assert outcome.exit_code != 0
    assert_conforms(
        out,
        [ExpectedDoc(rel_key="doc.pdf", pages=1, status="partial")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert doc_meta["status"] == "partial"
    assert "analysis failed" in (doc_meta.get("error") or "")


# ---------------------------------------------------------------------------
# SYS-02 (round-2): an unreadable input in the idempotency pre-check must NOT
# abort the batch — it is recorded as a per-file failure and the run continues.
# ---------------------------------------------------------------------------


def test_unreadable_input_records_failed_and_continues_batch(tmp_path, monkeypatch):
    """One unreadable file -> recorded status=failed, the other files still run.

    The pre-check uses safe_checksum, so an OSError on one input does not
    propagate and kill the whole batch (the SYS-02 'one bad file aborts the
    batch' failure mode)."""
    root = tmp_path / "in"
    root.mkdir()
    good = root / "good.pdf"
    bad = root / "bad.pdf"
    _make_pdf(good, pages=1)
    _make_pdf(bad, pages=1)
    out = tmp_path / "out"

    from ocr_output_contract import contract as _contract

    real_checksum = _contract.sha256_checksum

    def selective(path):
        if Path(path).name == "bad.pdf":
            raise PermissionError("permission denied (simulated)")
        return real_checksum(path)

    # Patch the symbol used by safe_checksum inside the contract module.
    monkeypatch.setattr(_contract, "sha256_checksum", selective)

    outcome = process(root, FakeBackend(), output_dir=out, show_progress=False)

    # The batch did NOT abort: the good file completed, the bad one is recorded.
    assert outcome.exit_code != 0
    assert (out / "good" / "good.md").exists()
    index = json.loads((out / "metadata.json").read_text())
    assert index["files"]["good.pdf"]["status"] == "completed"
    assert index["files"]["bad.pdf"]["status"] == "failed"
