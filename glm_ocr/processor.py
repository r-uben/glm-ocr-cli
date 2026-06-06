"""Document processing for GLM-OCR, routed through the canonical output contract.

glm is a *local, page-based* engine: it renders a PDF to per-page PIL images and
OCRs each page through one of its backends (ollama / vllm / transformers). This
module owns *how OCR happens* (rendering, the per-page backend calls, figure
extraction); the shared ``ocr-output-contract`` package owns *where the bytes go*
and what shape the metadata takes, so glm's output is byte-structure-identical to
every sibling engine.

The per-page text list this module produces is fed straight into the contract's
:func:`assemble_pages`, so every page of a document lands in ONE
``<root>/<rel/dir>/<stem>/<stem>.md`` under ``## Page N`` headers. Provenance
lives ONLY in the JSON sidecars (per-doc + root index) — the markdown body is
clean, with no YAML frontmatter (canon §3). Failures are recorded with
``status=failed``; partial documents with ``status=partial``; and any failure or
partial drives a nonzero exit via :class:`RunOutcome` (canon §6).
"""

from __future__ import annotations

import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    failure_checksum,
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    safe_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image
from tqdm import tqdm

from glm_ocr.config import settings
from glm_ocr.utils import collect_files, is_pdf_file, load_image

if TYPE_CHECKING:
    from glm_ocr.backends.base import Backend

logger = logging.getLogger(__name__)

#: Inline markdown image link ``![alt](target)`` — mirrors the conformance
#: harness's own pattern so the cached-skip side-output check enforces exactly
#: what :func:`ocr_output_contract.conformance.assert_conforms` would reject.
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _is_external_link(target: str) -> bool:
    """True for link targets that are NOT local files (URLs, data URIs, anchors)."""
    lowered = target.lower()
    return "://" in lowered or lowered.startswith(("http:", "https:", "data:", "mailto:", "#"))


def _markdown_side_outputs_intact(markdown_path: Path) -> bool:
    """True if every local inline image link in ``markdown_path`` resolves on disk.

    A cached skip trusts :meth:`RootIndex.is_completed`, which re-validates only
    the markdown body's existence — never the figure/image side outputs it
    references. So a figure PNG deleted (or never re-synced) after a completed
    run leaves a DANGLING inline link that the v0.1.3 conformance harness
    rejects, yet the doc would be skipped and exit 0. This re-derives the same
    local-image-link resolution the harness uses; a False result tells
    :func:`process` to reprocess instead of skip. A missing/unreadable markdown
    file also returns False (treat as needing reprocessing).
    """
    try:
        text = markdown_path.read_text(encoding="utf-8")
    except OSError:
        return False
    base = markdown_path.parent
    for raw in _IMAGE_LINK_RE.findall(text):
        target = raw.strip()
        # Strip an optional markdown title: ![](path "Title") / ![](path 'Title').
        for quote in ('"', "'"):
            qpos = target.find(f" {quote}")
            if qpos != -1:
                target = target[:qpos].strip()
                break
        # Strip surrounding angle brackets: ![](<path with spaces>).
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        if not target or _is_external_link(target):
            continue
        target = target.split("#", 1)[0].split("?", 1)[0]
        if not target:
            continue
        if not (base / target).resolve().exists():
            return False
    return True


@dataclass
class FigureInfo:
    """Container for one extracted figure (engine-owned; layout is the contract's)."""

    page_num: int
    figure_num: int
    image: Image.Image
    width: int
    height: int
    format: str
    context: str = ""
    description: str = ""
    saved_path: Path | None = None


@dataclass
class DocResult:
    """Result of OCR'ing one source document.

    ``pages`` holds the per-page markdown in order; ``page_errors`` maps a
    1-indexed page number to its error string for any page that failed (or whose
    model response was empty). ``figure_errors`` lists the failures from the
    opt-in figure path (a figure that could not be SAVED, or, under
    ``--analyze-figures``, a caption the model could not produce). ``status`` maps
    to the contract enum so a markdown full of error stubs — OR a doc whose
    explicitly-requested figures failed — can never be recorded as ``completed``.
    """

    source: Path
    pages: list[str] = field(default_factory=list)
    figures_md: str = ""
    processing_time: float = 0.0
    page_errors: dict[int, str] = field(default_factory=dict)
    figure_errors: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        """Status across BOTH the page-OCR channel and the figure channel.

        ``failed`` if a doc-level error left no pages, or no page succeeded;
        ``partial`` if some (but not all) pages succeeded, OR every page
        succeeded but a requested figure save/analysis failed (the failure must
        be VISIBLE in status/exit, never a silent ``completed``); ``completed``
        only when every page AND every requested figure succeeded.
        """
        if self.error is not None and not self.pages:
            return Status.FAILED
        if self.pages and not self.page_errors and self.error is None:
            # All pages clean — but a figure-channel failure still degrades the
            # doc to partial so the user who asked for figures is not told the
            # run "completed" while a figure was lost / could not be described.
            return Status.PARTIAL if self.figure_errors else Status.COMPLETED
        succeeded = self.page_count - len(self.page_errors)
        if succeeded > 0:
            return Status.PARTIAL
        return Status.FAILED


class OCRProcessor:
    """Renders + OCRs documents and routes all output through the contract."""

    def __init__(
        self,
        backend: Backend | None = None,
        output_dir: Path | None = None,
        extract_images: bool = False,
        dpi: int = 200,
        workers: int = 1,
        analyze_figures: bool = False,
        task: str = "text",
        raw: bool = False,
    ):
        if backend is not None:
            self._backend = backend
        else:
            from glm_ocr.backends import create_backend

            self._backend = create_backend()

        self.output_dir = output_dir
        self.extract_images = extract_images or settings.extract_images
        self.dpi = dpi
        self.workers = max(1, workers)
        self.analyze_figures = analyze_figures
        self.task = task
        self.raw = raw

    # ------------------------------------------------------------------
    # Rendering / figure extraction (engine-owned)
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        logger.debug(f"Converting PDF to images: {pdf_path} at {self.dpi} DPI")
        images: list[Image.Image] = []
        try:
            with fitz.open(pdf_path) as pdf_document:
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                for page_num in range(len(pdf_document)):
                    pix = pdf_document[page_num].get_pixmap(matrix=mat)
                    images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
                logger.info(f"Converted {len(images)} pages from {pdf_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF {pdf_path}: {e}") from e
        return images

    def _save_page_images(self, images: list[Image.Image], doc_dir: Path) -> None:
        """Save full-page rasters under ``<doc_dir>/images/`` (opt-in, no AI)."""
        images_dir = doc_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(images, 1):
            image.save(images_dir / f"page_{idx:04d}.png", "PNG")

    def _extract_figures_from_pdf(self, pdf_path: Path) -> tuple[list[FigureInfo], list[str]]:
        """Extract embedded raster figures from a PDF.

        Returns ``(figures, extraction_errors)``. Each per-image extraction is
        guarded independently so one undecodable image cannot abort the document
        (those are logged, not surfaced — a single bad embedded image is not a
        document-level failure of the requested channel). But a *wholesale*
        extraction crash (the ``fitz.open`` / page-walk body raising) is a
        failure of the entire requested figure channel: it is surfaced in
        ``extraction_errors`` so the caller can degrade the doc to ``partial``
        instead of silently recording ``completed``/exit 0 with an empty figure
        list (the HIGH "silent loss of the requested figure channel" bug).
        """
        figures: list[FigureInfo] = []
        extraction_errors: list[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    for img_idx, img_info in enumerate(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            base_image = doc.extract_image(xref)
                            pil_image = Image.open(io.BytesIO(base_image["image"]))
                            if pil_image.mode != "RGB":
                                pil_image = pil_image.convert("RGB")

                            context = ""
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                rect = img_rects[0]
                                expanded = fitz.Rect(
                                    max(0, rect.x0 - 50),
                                    max(0, rect.y0 - 150),
                                    rect.x1 + 50,
                                    rect.y1 + 150,
                                )
                                context = page.get_text("text", clip=expanded).strip()
                            if not context:
                                context = page_text[:500].strip() if page_text else ""

                            figures.append(
                                FigureInfo(
                                    page_num=page_num + 1,
                                    figure_num=img_idx + 1,
                                    image=pil_image,
                                    width=base_image["width"],
                                    height=base_image["height"],
                                    format=base_image["ext"],
                                    context=context,
                                )
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract image {img_idx + 1} from page "
                                f"{page_num + 1}: {e}"
                            )
            logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to extract figures from {pdf_path}: {e}")
            extraction_errors.append(f"figure extraction failed: {e}")
        return figures, extraction_errors

    def _save_figures(self, figures: list[FigureInfo], doc_dir: Path) -> list[str]:
        """Save figures as PNG under the canonical ``figures/figure_<N>_page<P>.png``.

        Each save is guarded so one undecodable image cannot abort the document;
        figures are always normalised to PNG (canon §4), so PyMuPDF native exts
        like ``jpx``/``jb2`` that PIL cannot write are never an issue.

        Returns the list of human-readable save-failure messages. A failure here
        is NOT swallowed silently: the caller folds these into the doc's
        ``figure_errors`` so the document is recorded ``partial`` (not a silent
        ``completed``) and the run exits nonzero — the user asked for figures and
        one was lost on disk.
        """
        figures_dir = figures_dir_for(doc_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for fig in figures:
            fig_path = figures_dir / figure_filename(fig.figure_num, fig.page_num)
            try:
                fig.image.save(fig_path, "PNG")
                fig.saved_path = fig_path
            except Exception as e:
                msg = f"figure {fig.figure_num} on page {fig.page_num}: save failed: {e}"
                logger.warning(msg)
                errors.append(msg)
        return errors

    def _analyze_single_figure(self, figure: FigureInfo) -> tuple[str, str | None]:
        """Analyze one figure. Returns (description, error)."""
        try:
            if figure.context:
                prompt = (
                    f"This figure appears in a document with the following context:\n"
                    f"---\n{figure.context[:500]}\n---\n\n"
                    f"Describe what this figure shows. Include details about any charts, "
                    f"graphs, diagrams, or visual elements. Explain what it represents "
                    f"in the context of the document."
                )
            else:
                prompt = (
                    "Describe this figure in detail. Include information about any charts, "
                    "graphs, diagrams, tables, or visual elements. Explain what it represents."
                )
            return self._backend.process_image(figure.image, prompt=prompt), None
        except Exception as e:
            logger.error(
                f"Failed to analyze figure {figure.figure_num} on page {figure.page_num}: {e}"
            )
            return "", str(e)

    def _analyze_figures(self, figures: list[FigureInfo], show_progress: bool = True) -> list[str]:
        """Populate figure descriptions in place; return analysis-failure messages.

        A caption the model could not produce is still embedded in the body as
        ``[Analysis Error: ...]`` (so the user sees it), AND surfaced in the
        returned list so the caller can fold it into ``figure_errors`` — a doc
        the user explicitly asked to ``--analyze-figures`` must NOT record
        ``completed``/exit 0 when its captions failed.
        """
        if not figures:
            return []
        errors: list[str] = []
        if self.workers == 1:
            iterator = (
                tqdm(figures, desc="Analyzing figures", unit="fig")
                if show_progress and len(figures) > 1
                else figures
            )
            for fig in iterator:
                desc, error = self._analyze_single_figure(fig)
                fig.description = desc if not error else f"[Analysis Error: {error}]"
                if error:
                    errors.append(
                        f"figure {fig.figure_num} on page {fig.page_num}: analysis failed: {error}"
                    )
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._analyze_single_figure, fig): fig for fig in figures
                }
                pbar = (
                    tqdm(
                        total=len(figures), desc=f"Analyzing figures ({self.workers}w)", unit="fig"
                    )
                    if show_progress and len(figures) > 1
                    else None
                )
                for future in as_completed(futures):
                    fig = futures[future]
                    desc, error = future.result()
                    fig.description = desc if not error else f"[Analysis Error: {error}]"
                    if error:
                        errors.append(
                            f"figure {fig.figure_num} on page {fig.page_num}: "
                            f"analysis failed: {error}"
                        )
                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.close()
        return errors

    def _figures_to_markdown(self, figures: list[FigureInfo]) -> str:
        """Render figure analyses as a trailing markdown section with canonical links."""
        saved = [f for f in figures if f.saved_path is not None]
        if not saved:
            return ""
        lines = ["\n\n---\n\n# Figures\n"]
        for fig in saved:
            lines.append(f"\n## Figure {fig.figure_num} (Page {fig.page_num})\n")
            lines.append(figure_markdown_link(fig.figure_num, fig.page_num) + "\n")
            lines.append(f"*Size: {fig.width}x{fig.height}*\n")
            if fig.description:
                lines.append(f"\n{fig.description}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-page OCR (bytes -> page text; output shape is the contract's job)
    # ------------------------------------------------------------------

    def _ocr_pages(
        self,
        images: list[Image.Image],
        prompt: str | None,
        show_progress: bool,
    ) -> tuple[list[str], dict[int, str]]:
        """OCR a list of page images, returning (page_texts, page_errors).

        Serial and parallel paths share ONE error policy (canon-aligned, fixes
        the audit's --workers-dependent semantics): a page that raises OR returns
        empty/whitespace text is recorded in ``page_errors`` and gets an explicit
        failure stub in its slot, keeping page numbering aligned.
        """
        page_errors: dict[int, str] = {}

        def ocr_one(idx: int, image: Image.Image) -> str:
            text = self._backend.process_image(
                image, prompt=prompt, task=self.task, return_raw=self.raw
            )
            if not text or not text.strip():
                raise ValueError("empty OCR response (no text returned)")
            return text

        pages: list[str] = [""] * len(images)
        if self.workers == 1:
            iterator = (
                tqdm(enumerate(images, 1), total=len(images), desc="OCR pages", unit="page")
                if show_progress and len(images) > 1
                else enumerate(images, 1)
            )
            for idx, image in iterator:
                try:
                    pages[idx - 1] = ocr_one(idx, image)
                except Exception as e:
                    logger.error(f"OCR failed for page {idx}: {e}")
                    page_errors[idx] = str(e)
                    pages[idx - 1] = f"*[OCR failed for page {idx}]*"
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(ocr_one, idx, image): idx for idx, image in enumerate(images, 1)
                }
                pbar = (
                    tqdm(total=len(images), desc=f"OCR pages ({self.workers}w)", unit="page")
                    if show_progress and len(images) > 1
                    else None
                )
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        pages[idx - 1] = future.result()
                    except Exception as e:
                        logger.error(f"OCR failed for page {idx}: {e}")
                        page_errors[idx] = str(e)
                        pages[idx - 1] = f"*[OCR failed for page {idx}]*"
                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.close()
        return pages, page_errors

    def process_file(
        self,
        file_path: Path,
        doc_dir: Path,
        prompt: str | None = None,
        show_progress: bool = True,
    ) -> DocResult:
        """OCR one file (PDF or image) into a DocResult. Never raises for OCR errors."""
        start = time.time()
        if not self._backend.model:
            self._backend.load_model()

        try:
            if is_pdf_file(file_path):
                images = self._pdf_to_images(file_path)
                if self.extract_images:
                    self._save_page_images(images, doc_dir)
                pages, page_errors = self._ocr_pages(images, prompt, show_progress)

                figures_md = ""
                figure_errors: list[str] = []
                if self.analyze_figures:
                    figures, extraction_errors = self._extract_figures_from_pdf(file_path)
                    # A WHOLESALE extraction crash (extraction_errors non-empty)
                    # must degrade the doc even when no figures came back — the
                    # user asked for the figure channel and it was lost. Folded
                    # in unconditionally so an empty figure list from a crash is
                    # never a silent completed/exit 0.
                    figure_errors.extend(extraction_errors)
                    if figures:
                        figure_errors.extend(self._save_figures(figures, doc_dir))
                        figure_errors.extend(
                            self._analyze_figures(figures, show_progress=show_progress)
                        )
                        figures_md = self._figures_to_markdown(figures)

                return DocResult(
                    source=file_path,
                    pages=pages,
                    figures_md=figures_md,
                    processing_time=time.time() - start,
                    page_errors=page_errors,
                    figure_errors=figure_errors,
                )

            # Single image input -> one page.
            image = load_image(file_path)
            pages, page_errors = self._ocr_pages([image], prompt, show_progress)
            return DocResult(
                source=file_path,
                pages=pages,
                processing_time=time.time() - start,
                page_errors=page_errors,
            )
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return DocResult(
                source=file_path, pages=[], processing_time=time.time() - start, error=str(e)
            )

    # ------------------------------------------------------------------
    # Output writing (all routed through the contract)
    # ------------------------------------------------------------------

    def fingerprint(self, prompt: str | None) -> str:
        """Run-config fingerprint for the cache key.

        Lets :meth:`RootIndex.is_completed` invalidate when any output-affecting
        run parameter changes, so a re-run under a different configuration
        reprocesses instead of silently reusing the prior output (the idempotency
        gap the audit flagged). Keyed on model / backend / task / prompt PLUS the
        engine's output-affecting flags via ``extra``:

        * ``raw`` — genuinely changes page text (``return_raw`` skips cleaning);
        * ``dpi`` / ``max_dim`` — change the rendered pixels fed to the model, so
          a re-run at a different resolution must re-OCR;
        * ``analyze_figures`` / ``extract_images`` — request new side outputs
          (figure captions / page rasters) that a prior run never produced.

        Without these in the key, a clean completed run would be silently skipped
        when re-run with ``--raw`` / a different ``--dpi``/``--max-dim`` / or
        ``--analyze-figures``/``--extract-images``, reusing stale markdown and
        never producing the newly requested outputs (exit 0).

        ``task`` is omitted when a custom ``prompt`` overrides it (the prompt then
        fully determines what the model is asked to do). All ``extra`` values are
        the RESOLVED effective flags (incl. falsy ones), so ``run_fingerprint``'s
        canonical JSON encoding keeps ``True``/``False`` distinct and key order
        irrelevant.
        """
        fp: str = run_fingerprint(
            model=self._backend.model_name,
            backend=self._backend.backend_name,
            task=None if prompt else self.task,
            prompt=prompt,
            extra={
                "raw": self.raw,
                "dpi": self.dpi,
                "max_dim": self._backend.max_dimension,
                "analyze_figures": self.analyze_figures,
                "extract_images": self.extract_images,
            },
        )
        return fp

    def _build_doc_metadata(
        self,
        result: DocResult,
        markdown_path: Path,
        output_root: Path,
        prompt: str | None,
    ) -> DocMetadata:
        status = result.status
        error = None
        if status is not Status.COMPLETED:
            if result.error:
                error = result.error
            else:
                # Surface BOTH channels so a figure-only failure (clean pages but
                # a save/analysis error) is never an empty diagnostic.
                parts = [f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items())]
                parts.extend(result.figure_errors)
                error = "; ".join(parts) if parts else None
        # Checksum the input WITHOUT raising: if the source became unreadable
        # (the SYS-02 path), still persist a status=failed record rather than
        # letting the failure-metadata write itself throw and abort the batch.
        # failure_checksum returns the real digest when readable, else the
        # canonical UNREADABLE_CHECKSUM sentinel (a VALID sha256: value the
        # conformance schema accepts, unlike the old "sha256:unreadable" stub),
        # which can never equal a real digest so a later readable run reprocesses.
        checksum = failure_checksum(result.source)
        return DocMetadata(
            status=status,
            checksum=checksum,
            model=self._backend.model_name,
            backend=self._backend.backend_name,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=str(markdown_path.relative_to(output_root)),
            pages=result.page_count,
            error=error,
            fingerprint=self.fingerprint(prompt),
        )

    def write_document(
        self,
        result: DocResult,
        output_root: Path,
        rel_key: str,
        doc_dir: Path,
        index: RootIndex,
        prompt: str | None = None,
    ) -> tuple[DocMetadata, Path]:
        """Write the aggregated markdown + BOTH metadata levels for one document.

        Output is always written (even on failure) so failures are recorded with
        ``status=failed`` per the canon. The single ``<stem>/<stem>.md``
        aggregates every page under ``## Page N`` headers with NO frontmatter.
        """
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        body = assemble_pages(result.pages) if result.pages else "*[OCR Failed]*\n"
        if result.figures_md:
            body = body + result.figures_md
        markdown_path.write_text(body, encoding="utf-8")

        meta = self._build_doc_metadata(result, markdown_path, output_root, prompt)
        write_doc_metadata(doc_dir, rel_key, meta)
        index.record(rel_key, meta)
        return meta, markdown_path


def process(
    source: Path,
    backend: Backend,
    output_dir: Path | None = None,
    *,
    prompt: str | None = None,
    task: str = "text",
    extract_images: bool = False,
    analyze_figures: bool = False,
    dpi: int = 200,
    workers: int = 1,
    reprocess: bool = False,
    raw: bool = False,
    show_progress: bool = True,
) -> RunOutcome:
    """Process an input (file or directory) through the canonical contract.

    Output goes to ``resolve_output_root(source, output_dir)`` — default
    ``<input-parent>/ocr/``; ``-o`` overrides; never required. Returns a
    :class:`RunOutcome` whose ``exit_code`` is nonzero if any document/page
    failed (uniform across single-file and batch, canon §6).

    The output root is resolved BEFORE discovery and passed into
    :func:`collect_files`, so the discovery walk excludes the resolved output
    subtree and never re-ingests the engine's own ``.md`` / figure / page-raster
    outputs as inputs on a re-run (the HIGH self-ingestion bug). Directory inputs
    are always walked recursively.
    """
    output_root = resolve_output_root(source, output_dir)

    files = collect_files(source, output_dir=output_root)
    if not files:
        raise ValueError(f"no supported files found at {source}")

    output_root.mkdir(parents=True, exist_ok=True)
    scan_root = source.parent if source.is_file() else source
    index = RootIndex(output_root)

    processor = OCRProcessor(
        backend=backend,
        output_dir=output_root,
        extract_images=extract_images,
        dpi=dpi,
        workers=workers,
        analyze_figures=analyze_figures,
        task=task,
        raw=raw,
    )

    # Run-config fingerprint: a doc completed under a different model/backend/
    # task/prompt is reprocessed (idempotency keyed on input AND run config).
    fingerprint = processor.fingerprint(prompt)

    outcome = RunOutcome()
    iterator = tqdm(files, desc="Processing files") if (show_progress and len(files) > 1) else files
    for file_path in iterator:
        rel_key = relative_key(file_path, scan_root)
        doc_dir = doc_dir_for(output_root, rel_key)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        # SYS-02: checksum the input WITHOUT aborting the batch. A file that
        # became unreadable between discovery and processing (permission denied,
        # deleted/replaced mid-run, broken symlink) yields None here; we record
        # that one document as status=failed and continue, rather than letting an
        # OSError propagate and kill the whole run.
        checksum = safe_checksum(file_path)
        if checksum is None:
            logger.error(f"failed to read {rel_key}: input unreadable")
            result = DocResult(source=file_path, error="input unreadable")
            meta, markdown_path = processor.write_document(
                result, output_root, rel_key, doc_dir, index, prompt=prompt
            )
            outcome.add(meta.status, detail=rel_key, output_path=str(markdown_path))
            continue

        if not reprocess and index.is_completed(rel_key, checksum, fingerprint=fingerprint):
            # is_completed re-validates only the markdown body, not the figure /
            # image side outputs the body links to. A deleted/dangling figure
            # leaves a body referencing a missing file (which the v0.1.3
            # conformance harness rejects), so verify those side outputs are
            # intact before honoring the skip; otherwise fall through and
            # reprocess to regenerate them.
            if _markdown_side_outputs_intact(markdown_path):
                logger.info(f"skip {rel_key} (already completed; use --reprocess)")
                # Emit the existing .md path even on a cached skip so quiet-mode
                # scripting (`glm-ocr dir -q | xargs ...`) still sees every doc.
                outcome.add(Status.COMPLETED, output_path=str(markdown_path))
                continue
            logger.warning(
                f"reprocess {rel_key}: cached output references a missing figure/image "
                f"side output (dangling inline link); regenerating"
            )

        result = processor.process_file(
            file_path, doc_dir, prompt=prompt, show_progress=show_progress
        )
        meta, markdown_path = processor.write_document(
            result, output_root, rel_key, doc_dir, index, prompt=prompt
        )
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

    return outcome
