# Build a Standalone Generic Embedding Service

## Objective

Create a **standalone, reusable local embedding service** that exposes a simple HTTP API for generating text embeddings.

The service must use **Google EmbeddingGemma 300M** running locally through **LiteRT**.

This is a **generic infrastructure/service project**.

Do NOT specialize it for filmmaking, OneShot, RAG, search, recommendations, or any other specific application.

Any external application should be able to send text to this service and receive embedding vectors.

---

# 1. Model

The service must use the locally downloaded EmbeddingGemma model.

Model files are located outside this project:

```text
D:\LLM\liteRT\embeddinggemma-300m\
├── embeddinggemma-300M_seq2048_mixed-precision.tflite
└── sentencepiece.model
```

Do NOT copy these files into the project.

The service should reference the model through configuration/environment variables.

Default paths may be:

```text
EMBEDDING_MODEL_PATH=D:\LLM\liteRT\embeddinggemma-300m\embeddinggemma-300M_seq2048_mixed-precision.tflite

TOKENIZER_PATH=D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model
```

However, these paths must be configurable.

Do not hardcode them throughout the source code.

---

# 2. Runtime

Use:

* Python
* FastAPI
* LiteRT via `ai-edge-litert`
* SentencePiece
* NumPy
* Pydantic
* Uvicorn
* `uv` for project/environment/dependency management

Python environment should be managed with `uv`.

Do not use Conda/Mamba for this project.

Do not use TensorFlow unless it is absolutely required.

Do not use PyTorch.

Do not use SentenceTransformers.

The service should directly use:

```python
from ai_edge_litert.interpreter import Interpreter
```

to execute the `.tflite` model.

---

# 3. Known Model Interface

The downloaded model has already been tested successfully.

LiteRT reports:

```text
Input:
name: embed_2048_text_batch:0
shape: [1, 2048]
dtype: int32

Output:
name: StatefulPartitionedCall:0
shape: [1, 768]
dtype: float32
```

The tokenizer has also been tested successfully using:

```text
sentencepiece.model
```

Therefore:

```text
text
  ↓
SentencePiece tokenizer
  ↓
token IDs
  ↓
prepare [1, 2048] int32 input
  ↓
LiteRT EmbeddingGemma
  ↓
[1, 768] float32 embedding
```

Preserve the model's expected input/output behavior.

Do not blindly change the tokenization or preprocessing logic.

If the implementation requires model-specific preprocessing details that are not known, isolate that logic in a dedicated component rather than spreading assumptions throughout the application.

---

# 4. Core Design Principle

The service must hide all model-specific implementation details.

A client should NOT need to know about:

* LiteRT
* TFLite
* SentencePiece
* token IDs
* 2048 token input
* tensor names
* tensor indices
* padding
* model files
* NumPy

The client should only interact with a clean API such as:

```text
POST /v1/embeddings
```

Example:

```json
{
  "input": "A camera slowly moves toward the subject."
}
```

Response:

```json
{
  "embedding": [0.0123, -0.0831, ...],
  "dimension": 768,
  "model": "embeddinggemma-300m"
}
```

---

# 5. API Design

Implement a versioned REST API.

Base path:

```text
/v1
```

## POST /v1/embeddings

Support a single string.

Request:

```json
{
  "input": "Hello world"
}
```

Response:

```json
{
  "object": "embedding",
  "embedding": [ ... ],
  "dimension": 768,
  "model": "embeddinggemma-300m"
}
```

---

## Batch embeddings

The same endpoint should also support multiple strings.

Request:

```json
{
  "input": [
    "Hello world",
    "This is another sentence.",
    "A third piece of text."
  ]
}
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "embedding": [ ... ]
    },
    {
      "index": 1,
      "embedding": [ ... ]
    },
    {
      "index": 2,
      "embedding": [ ... ]
    }
  ],
  "model": "embeddinggemma-300m",
  "dimension": 768
}
```

Preserve input ordering.

The API should process batches efficiently rather than requiring the client to make one HTTP request per text.

---

# 6. Health Endpoint

Implement:

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

This endpoint should be lightweight.

---

# 7. Model Information Endpoint

Implement:

```text
GET /v1/models
```

Return useful information such as:

```json
{
  "model": "embeddinggemma-300m",
  "dimension": 768,
  "max_sequence_length": 2048,
  "runtime": "litert"
}
```

Do not expose unnecessary internal filesystem information.

---

# 8. Architecture

Use a clean layered architecture.

Suggested structure:

```text
embedding-service/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_embeddings.py
│   │   └── routes_health.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── embedding_service.py
│       ├── tokenizer.py
│       └── litert_engine.py
│
├── tests/
│   ├── test_health.py
│   ├── test_embeddings.py
│   └── test_tokenizer.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

You may modify the structure if there is a strong technical reason, but maintain clear separation between:

1. API layer
2. configuration
3. tokenizer
4. LiteRT model execution
5. embedding service/business logic
6. tests

---

# 9. Model Loading

The model should be loaded **once when the application starts**.

Do NOT load the `.tflite` model for every API request.

Bad:

```text
request
 ↓
load model
 ↓
run model
 ↓
response
```

Correct:

```text
application startup
 ↓
load tokenizer
 ↓
load LiteRT model
 ↓
allocate tensors
 ↓
service stays alive
 ↓
requests reuse loaded model
```

Use FastAPI's modern lifespan mechanism for initialization/shutdown.

---

# 10. Thread Safety / Concurrency

The LiteRT interpreter should not be assumed to be safely callable concurrently from multiple requests.

Design the inference layer so concurrent HTTP requests cannot corrupt interpreter state.

Possible approaches include:

* an inference lock
* a controlled worker model
* batching/queueing

Choose a simple reliable approach first.

Do not over-engineer distributed inference.

This is intended to be a **local service**.

---

# 11. Configuration

Use environment variables.

Example `.env.example`:

```env
HOST=127.0.0.1
PORT=8100

EMBEDDING_MODEL_PATH=D:\LLM\liteRT\embeddinggemma-300m\embeddinggemma-300M_seq2048_mixed-precision.tflite
TOKENIZER_PATH=D:\LLM\liteRT\embeddinggemma-300m\sentencepiece.model

MODEL_NAME=embeddinggemma-300m
EMBEDDING_DIMENSION=768
MAX_SEQUENCE_LENGTH=2048
```

Use Pydantic settings or an equivalent clean configuration mechanism.

Never hardcode machine-specific paths in application logic.

---

# 12. Error Handling

Return proper HTTP errors.

Examples:

### Missing model

```text
503 Service Unavailable
```

### Invalid request

```text
400 Bad Request
```

### Empty input

Reject empty strings.

### Model inference failure

Return an appropriate `500` response without exposing Python stack traces or internal filesystem details to the client.

Log the actual exception server-side.

---

# 13. Input Validation

Validate:

* string input is not empty
* batch input is not empty
* batch size has a reasonable configurable limit
* text length is handled safely
* malformed JSON is rejected automatically by FastAPI/Pydantic

Do not silently truncate text without documenting the behavior.

Because the model has a 2048-token input limit, handle long inputs deliberately.

For example, either:

1. reject inputs exceeding the supported limit, or
2. expose a configurable truncation policy.

Do not silently invent a chunking/RAG system.

This service is an embedding service, not a document-processing system.

---

# 14. Embedding Output

The model produces:

```text
[1, 768]
```

Return the embedding as a normal JSON array of floats.

Do not expose NumPy objects directly.

Convert to standard Python types:

```python
embedding.tolist()
```

The service should guarantee that every successful embedding has:

```text
dimension = 768
```

---

# 15. Normalization

Do NOT silently normalize embeddings unless the behavior is explicitly configurable.

If useful, provide an optional request parameter such as:

```json
{
  "input": "Hello world",
  "normalize": true
}
```

If implemented, document exactly what normalization means.

Default behavior should preserve the raw model output.

---

# 16. API Documentation

FastAPI's automatic OpenAPI documentation should work.

The service should expose:

```text
/docs
```

and:

```text
/openapi.json
```

Use clear request/response schemas.

---

# 17. Logging

Provide clean logs such as:

```text
INFO - Loading EmbeddingGemma model
INFO - Model loaded successfully
INFO - Embedding dimension: 768
INFO - Server started on 127.0.0.1:8100
```

Do not log the full contents of user-provided text by default.

Avoid logging embeddings.

---

# 18. Testing

Create tests for:

### Health

```text
GET /health
→ 200
```

### Single embedding

```text
POST /v1/embeddings
→ 768-dimensional vector
```

### Batch embedding

```text
3 inputs
→ 3 vectors
→ correct ordering
```

### Empty input

```text
→ validation error
```

### Model loading

Verify that the model and tokenizer can be loaded.

### Determinism

The same input should produce the same embedding within expected floating-point tolerance.

### Similarity sanity test

Create a small test comparing:

```text
"A dog is running in a park."
"A dog runs through a park."
```

against:

```text
"The database server is unavailable."
```

Do not hardcode arbitrary similarity thresholds unless they are experimentally justified. The purpose is only to verify that inference behaves sensibly.

---

# 19. CLI

Also provide a simple CLI/test command so the service does not have to be started just to test the model.

For example:

```powershell
uv run python -m app.cli "Hello world"
```

Output:

```text
Model: embeddinggemma-300m
Dimension: 768
Embedding:
[...]
```

This should use the same underlying `EmbeddingService` used by the API.

Do not duplicate model inference logic.

---

# 20. Client Examples

The README must contain examples for at least:

### cURL

```bash
curl -X POST http://127.0.0.1:8100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"input\":\"Hello world\"}"
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

The service must therefore remain completely language-agnostic.

---

# 21. Security

Default binding should be:

```text
127.0.0.1
```

not:

```text
0.0.0.0
```

This is a local AI service.

Do not add authentication unless there is a concrete need.

However, structure the application so authentication could be added later.

---

# 22. No Application-Specific Logic

This is extremely important.

DO NOT include:

* OneShot
* filmmaking
* screenplay analysis
* camera shots
* cinematography
* RAG
* FAISS
* Qdrant
* vector database
* document ingestion
* chunking pipelines
* search logic
* recommendation logic
* application-specific prompts

The service's only responsibility is:

```text
TEXT → EMBEDDING VECTOR
```

Everything else belongs to the consuming application.

---

# 23. No Model Duplication

Do not copy:

```text
embeddinggemma-300M_seq2048_mixed-precision.tflite
```

or:

```text
sentencepiece.model
```

into the repository.

The repository should remain lightweight.

The model location must be configurable.

---

# 24. Developer Experience

The project must be runnable with:

```powershell
uv sync
```

and then:

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Also provide a convenient script/command if appropriate.

The README must explain:

1. Requirements
2. Model files
3. Model path configuration
4. Installation
5. Running the server
6. API usage
7. Batch usage
8. CLI usage
9. Testing
10. Configuration
11. Architecture
12. Troubleshooting

---

# 25. Expected Final Architecture

The final system should look conceptually like:

```text
                    HTTP CLIENTS
                         │
             ┌───────────┼───────────┐
             │           │           │
          OneShot     Project B   Project C
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Embedding Service  │
              │                     │
              │      FastAPI        │
              │         │           │
              │  EmbeddingService   │
              │         │           │
              │   ┌─────┴─────┐     │
              │   │           │     │
              │Tokenizer   LiteRT   │
              │   │           │     │
              │   └─────┬─────┘     │
              │         │           │
              └─────────┼───────────┘
                        │
                        ▼
               EmbeddingGemma 300M
                        │
                        ▼
                  768-d vector
```

---

# 26. Important Implementation Rule

Before writing code, inspect the project requirements and current dependency versions.

Do not blindly assume APIs.

Use the currently installed:

```text
ai-edge-litert
```

package and its actual Python API.

The implementation should remain compatible with the local model:

```text
embeddinggemma-300M_seq2048_mixed-precision.tflite
```

and:

```text
sentencepiece.model
```

If an implementation detail is uncertain, investigate it rather than inventing an API.

---

# Final Goal

At the end, I should be able to do:

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8100
```

and then from **any application**:

```http
POST http://127.0.0.1:8100/v1/embeddings
```

with:

```json
{
  "input": "Any text from any application."
}
```

and receive:

```json
{
  "object": "embedding",
  "embedding": [768 floating-point values],
  "dimension": 768,
  "model": "embeddinggemma-300m"
}
```

The resulting project should be **generic, modular, documented, testable, locally runnable, and reusable by completely unrelated applications.**
