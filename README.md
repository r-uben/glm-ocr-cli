# glm-ocr-cli

CLI tool for OCR using [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B) by Zhipu AI via Ollama or vLLM.

## Install

```bash
pip install glm-ocr-cli
# For vLLM backend:
pip install 'glm-ocr-cli[vllm]'
```

## Prerequisites

Pull the model in Ollama:

```bash
ollama pull glm-ocr
```

## Usage

```bash
glm-ocr document.pdf
glm-ocr ./papers/ --recursive
glm-ocr image.png --task formula
glm-ocr paper.pdf --dry-run
```

### Task types

| Task | Prompt | Output |
|------|--------|--------|
| `text` | Text Recognition | Markdown |
| `formula` | Formula Recognition | LaTeX |
| `table` | Table Recognition | Markdown tables |
| `figure` | Figure Recognition | Description |

### Backends

- **Ollama** (default): `glm-ocr document.pdf`
- **vLLM**: `glm-ocr document.pdf --backend vllm --vllm-url http://localhost:8080/v1`

## Output

Output writing follows the shared [`ocr-output-contract`](https://github.com/r-uben/ocr-output-contract),
so glm's output is byte-structure-identical to every sibling OCR engine:

- Default output root is `<input-parent>/ocr/`; `-o`/`--output-dir` overrides it but is never required.
- One aggregated `<root>/<rel/dir>/<stem>/<stem>.md` per document, mirroring the input subtree (so
  same-basename files in different folders never collide), with every page under a `## Page N` header.
- The markdown body is **clean** — no YAML frontmatter. All provenance lives in `metadata.json`
  sidecars at two levels: a per-document `<stem>/metadata.json` and a rolled-up root index keyed by
  input-relative path (`{status, checksum, model, backend, processing_time, timestamp, output_path, pages}`).
- Failures are recorded with `status="failed"` (partial documents `status="partial"`), and the process
  exits nonzero if any document or page failed.

## Configuration

Environment variables (prefix `GLM_OCR_`):

```bash
export GLM_OCR_BACKEND=ollama
export GLM_OCR_MODEL_NAME=glm-ocr
export GLM_OCR_OLLAMA_URL=http://localhost:11434
export GLM_OCR_VLLM_BASE_URL=http://localhost:8080/v1
```

## License

MIT
