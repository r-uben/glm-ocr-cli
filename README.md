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
