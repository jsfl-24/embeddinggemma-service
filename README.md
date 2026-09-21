# embeddinggemma-service

Generic local text embedding HTTP service using **Google EmbeddingGemma 300M** running through **LiteRT** (`ai-edge-litert`), SentencePiece, and NumPy.

The service exposes a model-agnostic API: clients send text and receive 768-dimensional embeddings. No LiteRT, TFLite, tokenizer, or model internals are exposed.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- The EmbeddingGemma model files (download them separately; they are **not** bundled with this repo)

## Model files

The service runs against a locally downloaded LiteRT model. The expected files:

```text
D:\LLM\liteRT\embeddinggemma-300m\
├── embeddinggemma-300M_seq2048_mixed-precision.tflite
└── sentencepiece.model
```

Model interface preserved by this service:

```text
input:  [1, 2048] int32    (token IDs, BOS + tokens + EOS + PAD)
output: [1, 768]  float32  (embedding)
```

## Model path configuration

Paths are read from environment variables (or a `.env` file) and can point anywhere on disk. Defaults:

```env
EMBEDDING_MODEL_PATH=D:\LLM\liteRT\embeddinggemma-300m\embeddinggemma-300M_seq2048_mixed-precision.tflite
TOKENIZER_PATH=D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model
```

Copy `.env.example` to `.env` and adjust if your files live elsewhere. The model files are never copied into this repository.

## Installation

The package is split into a lightweight client (only `httpx`) and the service runtime. To develop/run the server:

```powershell
uv sync --extra server --extra dev
```

Other applications that only consume embeddings should install the base package (see [Using the client from another application](#using-the-client-from-another-application)).

## Running the server

```powershell
uv run uvicorn embeddinggemma_service.main:app --host 127.0.0.1 --port 8100
```

The model and tokenizer are loaded once at startup and reused for every request. Server binding defaults to `127.0.0.1` (change with `HOST`/`PORT`).

## API usage

Interactive docs are available at `http://127.0.0.1:8100/docs` (OpenAPI at `/openapi.json`).

### cURL

```bash
curl -X POST http://127.0.0.1:8100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world"}'
```

Response:

```json
{
  "object": "embedding",
  "embedding": [0.0123, -0.0831, ...],
  "dimension": 768,
  "model": "embeddinggemma-300m"
}
```

### Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8100/v1/embeddings",
    json={"input": "Hello world"}
)

embedding = response.json()["embedding"]
```

### JavaScript

```javascript
const response = await fetch(
    "http://127.0.0.1:8100/v1/embeddings",
    {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            input: "Hello world"
        })
    }
);

const data = await response.json();
```

## Using the client from another application

The same package ships a lightweight HTTP client (`httpx` only — no LiteRT, SentencePiece, or model files) so consuming applications only see `text → embedding`:

```python
from embeddinggemma_service.client import EmbeddingClient

client = EmbeddingClient()  # defaults to http://127.0.0.1:8100
with client:
    embedding = client.embed("Hello world")          # -> list[float]
    embeddings = client.embed_batch(["Hello", "world"])  # -> list[list[float]]
```

- Add it to your project with `uv add embeddinggemma-service` — this pulls only `httpx`.
- Point it elsewhere via `EmbeddingClient(base_url=...)` or `EMBEDDING_SERVICE_URL` (used by `EmbeddingClient.from_env()`).
- Pass `normalize=True` to L2-normalize.
- One client instance is thread-safe and can be reused for all embedding calls.
- The `Embedder` protocol (`embed` / `embed_batch`) documents the interface application code should rely on so the underlying implementation can be swapped without touching callers.
- Errors: `EmbeddingServiceUnavailable` (service down/503/5xx) and `InvalidEmbeddingInput` (400); both subclass `EmbeddingClientError`.

## Batch usage

Send a list of strings; the response preserves input order.

```json
{
  "input": [
    "Hello world",
    "This is another sentence.",
    "A third piece of text."
  ]
}
```

```json
{
  "object": "list",
  "data": [
    {"index": 0, "embedding": [...]},
    {"index": 1, "embedding": [...]},
    {"index": 2, "embedding": [...]}
  ],
  "model": "embeddinggemma-300m",
  "dimension": 768
}
```

Each text is embedded in declaration order, internally processed one model invocation per text (the exported model has a fixed `[1, 2048]` input).

### Normalization

Optional per-request L2 normalization:

```json
{"input": "Hello world", "normalize": true}
```

Defaults to `false`, which returns the raw model output.

### Other endpoints

- `GET /health` — liveness, returns `{"status": "ok"}`
- `GET /v1/models` — model metadata (name, dimension, max sequence length, runtime)

## CLI usage

Generate an embedding without starting the server (requires the `server` extra):

```powershell
uv run python -m embeddinggemma_service.cli "Hello world"
```

Or via the installed script:

```powershell
uv run embeddinggemma-service "Hello world"
```

Output:

```text
Model: embeddinggemma-300m
Dimension: 768
Embedding:
[-0.16002678871154785, 0.028115276247262955, ...]
```

Add `--normalize` to L2-normalize. The CLI uses the same `EmbeddingService` as the API.

## Testing

```powershell
uv run pytest
```

Tests cover health, single/batch embeddings, ordering, determinism, normalization, input validation, tokenizer behavior (special tokens, padding, truncation), similarity sanity, and the HTTP client (via a mock transport, with no model required). Tests that need model inference require the local model files and are skipped when they are missing.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8100` | Bind port |
| `EMBEDDING_MODEL_PATH` | `D:\LLM\...\embeddinggemma-300M_seq2048_mixed-precision.tflite` | Path to the `.tflite` model |
| `TOKENIZER_PATH` | `D:\LLM\...\sentencepiece.model` | Path to the SentencePiece tokenizer |
| `MODEL_NAME` | `embeddinggemma-300m` | Model name returned by the API |
| `EMBEDDING_DIMENSION` | `768` | Expected output dimension (validated) |
| `MAX_SEQUENCE_LENGTH` | `2048` | Fixed model input length |
| `MAX_BATCH_SIZE` | `32` | Maximum texts per batch request |
| `TRUNCATION_MODE` | `truncate` | `truncate` silently truncates over-long inputs to 2048 tokens; `error` rejects them with 400 |
| `NUM_THREADS` | `4` | LiteRT CPU threads |

## Architecture

```text
HTTP client
    │
    ▼
FastAPI (api/)
    │
    ▼
EmbeddingService (services/embedding_service.py)
    │
    ├── TokenizerService (services/tokenizer.py)  — SentencePiece + input shaping
    └── LiteRTEngine (services/litert_engine.py)  — Interpreter + inference lock
                        │
                        ▼
                 EmbeddingGemma 300M (.tflite)
                        │
                        ▼
                   768-d embedding
```

- `api/` — HTTP routes and request/response schemas
- `core/` — configuration (`config.py`) and logging
- `models/` — Pydantic request/response schemas
- `services/` — tokenizer, LiteRT execution, and embedding business logic
- Model loading and cleanup happen in the FastAPI lifespan; the interpreter is reused across requests and guarded by a lock so concurrent requests cannot corrupt interpreter state.

## Troubleshooting

- **Model is not loaded / `503` on `/v1/embeddings`** — check `EMBEDDING_MODEL_PATH` and `TOKENIZER_PATH`, and that the files exist. The server still starts and serves `/health`; the failure is logged at startup.
- **Slow startup** — the ~220 MB model loads once at startup (a few dozen seconds on CPU); requests then reuse it.
- **`VIRTUAL_ENV` mismatch warning from uv** — a stale `VIRTUAL_ENV` value is present in your shell; start a fresh shell or unset it so `uv` targets this project's `.venv`.
- **Validation errors return `400`** — empty strings, empty batches, oversized batches, and (in `error` truncation mode) over-length inputs are rejected as invalid requests.
- **Inference failures return `500`** — check the server logs; the client only receives a generic message.