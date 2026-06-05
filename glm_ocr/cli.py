"""Command-line interface for GLM-OCR.

Output writing is owned by the shared ``ocr-output-contract`` package: output
goes to ``<input-parent>/ocr/`` by default (``-o`` overrides; never required),
aggregated as one ``<stem>/<stem>.md`` per document under ``## Page N`` headers
with NO YAML frontmatter, plus per-doc + root ``metadata.json`` sidecars. The
exit code is nonzero if any document/page failed (canon §6).
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from glm_ocr import __version__
from glm_ocr.backends import create_backend
from glm_ocr.config import settings
from glm_ocr.processor import process
from glm_ocr.utils import collect_files, is_pdf_file, setup_logging

console = Console()
err_console = Console(stderr=True)


def print_banner(quiet: bool = False) -> None:
    if not quiet:
        console.print(f"[dim]glm-ocr v{__version__}[/dim]")


def _format_size(size_bytes: float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_pdf_page_count(path: Path) -> int:
    """Get page count from a PDF without rendering."""
    import fitz

    with fitz.open(path) as doc:
        return len(doc)


def _run_dry_run(input_path: Path, recursive: bool, quiet: bool) -> None:
    """List files that would be processed without actually processing them."""
    files = collect_files(input_path, recursive=recursive)

    if quiet:
        for f in files:
            console.print(str(f))
        return

    table = Table(title="Files to process (dry run)", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Pages", justify="right")

    total_size = 0
    total_pages = 0

    for idx, f in enumerate(files, 1):
        size = f.stat().st_size
        total_size += size

        if is_pdf_file(f):
            try:
                pages = _get_pdf_page_count(f)
            except Exception:
                pages = 0
            file_type = "PDF"
        else:
            pages = 1
            file_type = f.suffix.upper().lstrip(".")

        total_pages += pages
        table.add_row(str(idx), f.name, file_type, _format_size(size), str(pages))

    console.print(table)
    console.print(
        f"\n[bold]{len(files)}[/bold] files, "
        f"[bold]{_format_size(total_size)}[/bold] total, "
        f"[bold]{total_pages}[/bold] pages"
    )


@click.group()
@click.version_option(version=__version__, prog_name="glm-ocr")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """GLM-OCR CLI - OCR processing via Ollama, vLLM, or local transformers.

    Process documents and images directly:

    \b
        glm-ocr document.pdf
        glm-ocr ./papers/ --recursive
        glm-ocr document.pdf --dry-run

    Uses GLM-OCR (0.9B) by Zhipu AI. Default backend is Ollama (local).
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output root (default: <input-parent>/ocr/). Writes <stem>/<stem>.md per document.",
)
@click.option("-r", "--recursive", is_flag=True, help="Recursively process directories")
@click.option(
    "--model",
    "model_name",
    type=str,
    default=None,
    help="Model name (default: settings.model_name / GLM_OCR_MODEL_NAME, else glm-ocr).",
)
@click.option("--prompt", type=str, help="Custom prompt for OCR (overrides --task).")
@click.option(
    "--task",
    type=click.Choice(["text", "formula", "table", "figure"]),
    default="text",
    show_default=True,
    help="OCR task type (selects the model prompt).",
)
@click.option("--extract-images", is_flag=True, help="Save full-page rasters under images/.")
@click.option(
    "--analyze-figures",
    is_flag=True,
    help="Extract and describe embedded figures from PDFs with AI captions.",
)
@click.option("--dpi", type=int, default=200, show_default=True, help="PDF rendering DPI.")
@click.option(
    "-w", "--workers", type=int, default=1, help="Parallel workers for PDF pages (default: 1)."
)
@click.option(
    "--max-dim",
    "max_dimension",
    type=int,
    default=None,
    help="Maximum image dimension (width or height). Default: 2048. Set to 0 to disable.",
)
@click.option(
    "--backend",
    type=click.Choice(["transformers", "ollama", "vllm"]),
    default=None,
    help="Backend: 'ollama' (local, default), 'transformers', or 'vllm'. Env: GLM_OCR_BACKEND.",
)
@click.option(
    "--vllm-url",
    "vllm_base_url",
    type=str,
    default=None,
    help="vLLM API URL (default: http://localhost:8080/v1). Env: GLM_OCR_VLLM_BASE_URL.",
)
@click.option("--reprocess", is_flag=True, help="Force re-OCR of already-completed files.")
@click.option(
    "--dry-run", is_flag=True, help="List files that would be processed without running OCR."
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress non-error output. Print output .md paths only (useful for scripting).",
)
@click.option("--verbose", is_flag=True, help="Enable verbose (DEBUG) logging.")
@click.pass_context
def process_cmd(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path | None,
    recursive: bool,
    model_name: str | None,
    prompt: str | None,
    task: str,
    extract_images: bool,
    analyze_figures: bool,
    dpi: int,
    workers: int,
    max_dimension: int | None,
    backend: str | None,
    vllm_base_url: str | None,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Process documents and images with OCR.

    INPUT_PATH can be a single file or a directory containing multiple files.

    Supported formats: PDF, JPG, PNG, WEBP, GIF, BMP, TIFF

    \b
    Examples:
        glm-ocr document.pdf
        glm-ocr ./documents/ --recursive
        glm-ocr paper.pdf --dry-run
        glm-ocr image.jpg --task formula
        glm-ocr paper.pdf -q | xargs ls -la
    """
    setup_logging(level=settings.log_level, verbose=verbose)

    if dry_run:
        print_banner(quiet=quiet)
        try:
            _run_dry_run(input_path, recursive=recursive, quiet=quiet)
        except Exception as e:
            err_console.print(f"[red]error:[/red] {e}")
            sys.exit(1)
        return

    print_banner(quiet=quiet)

    backend_type = backend or settings.backend
    # --model default=None so GLM_OCR_MODEL_NAME / settings.model_name take effect.
    resolved_model = model_name or settings.model_name

    try:
        backend_instance = create_backend(
            backend_type=backend_type,
            model_name=resolved_model,
            max_dimension=max_dimension,
            ollama_url=settings.ollama_url,
            vllm_base_url=vllm_base_url or settings.vllm_base_url,
        )
        backend_instance.load_model()

        outcome = process(
            input_path,
            backend_instance,
            output_dir=output_dir,
            recursive=recursive,
            prompt=prompt,
            task=task,
            extract_images=extract_images,
            analyze_figures=analyze_figures,
            dpi=dpi,
            workers=workers,
            reprocess=reprocess,
            show_progress=not verbose and not quiet,
        )

        backend_instance.unload_model()
    except Exception as e:
        err_console.print(f"[red]error:[/red] {e}")
        sys.exit(1)

    total = outcome.completed + outcome.failed + outcome.partial
    if quiet:
        for path in outcome.outputs:
            console.print(path)
    else:
        console.print(f"[dim]{outcome.completed}/{total} document(s) completed[/dim]")
        if outcome.has_failures:
            err_console.print(
                f"[yellow]{outcome.failed} failed, {outcome.partial} partial:[/yellow] "
                + ", ".join(outcome.failures)
            )

    # Uniform exit policy (canon §6): nonzero if any document/page failed.
    if outcome.exit_code != 0:
        sys.exit(outcome.exit_code)


# socr's BaseEngine calls the subcommand `process`; expose it under that name.
cli.add_command(process_cmd, name="process")


@cli.command()
def info() -> None:
    """Show system and configuration information."""
    print_banner()

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Status", style="green")

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    sys_table.add_row("Backend", settings.backend)

    # Only probe Ollama when it is the active backend (avoid a 5s timeout otherwise).
    if settings.backend == "ollama":
        from glm_ocr.backends.ollama import OllamaBackend

        ob = OllamaBackend()
        running = ob._check_ollama_running()
        model_ok = ob._check_model_available() if running else False
        sys_table.add_row("Ollama URL", settings.ollama_url)
        sys_table.add_row("Ollama Running", "Yes" if running else "No")
        sys_table.add_row("glm-ocr Model", "Available" if model_ok else "Not found")
    elif settings.backend == "vllm":
        sys_table.add_row("vLLM URL", settings.vllm_base_url)

    console.print(sys_table)

    settings_table = Table(title="Current Settings")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", style="yellow")
    settings_table.add_row("Model", settings.model_name)
    settings_table.add_row("Default Output", "<input-parent>/ocr/")
    settings_table.add_row("Max Image Dimension", str(settings.max_dimension))
    console.print(settings_table)

    console.print("\n[bold]Supported Formats:[/bold]")
    console.print("Images: JPG, PNG, WEBP, GIF, BMP, TIFF")
    console.print("Documents: PDF\n")

    console.print("[bold]Backend Options:[/bold]")
    console.print("  --backend ollama        Ollama server (local, default)")
    console.print("  --backend transformers  Local HuggingFace inference")
    console.print("  --backend vllm          OpenAI-compatible API (GPU server)\n")

    console.print("[bold]Task Types:[/bold]")
    console.print("  --task text       General text recognition (default)")
    console.print("  --task formula    LaTeX formula extraction")
    console.print("  --task table      Markdown table extraction")
    console.print("  --task figure     Figure description\n")


def main() -> None:
    """Entry point. Auto-inserts 'process' when first arg is a file/directory path."""
    argv = sys.argv[1:]

    if argv:
        known_subcommands = {"process", "info"}

        first_non_option_index = None
        for idx, arg in enumerate(argv):
            if not arg.startswith("-"):
                first_non_option_index = idx
                break

        if first_non_option_index is not None:
            candidate = argv[first_non_option_index]
            if candidate not in known_subcommands and Path(candidate).exists():
                argv = argv[:first_non_option_index] + ["process"] + argv[first_non_option_index:]
                sys.argv = [sys.argv[0], *argv]

    cli(obj={})


if __name__ == "__main__":
    main()
