# AGENTS.md

## Project

Generic local text embedding HTTP service using EmbeddingGemma 300M + LiteRT.

Read `SPEC.md` before implementing or changing architecture.

## Rules

* Keep this project generic. No OneShot, filmmaking, RAG, vector DB, or application-specific logic.
* Use Python + FastAPI + LiteRT (`ai-edge-litert`) + SentencePiece + NumPy.
* Use `uv` for environment and dependency management.
* Model files are external; never copy them into the repository.
* Keep model paths configurable through environment variables.
* Load tokenizer and LiteRT model once at application startup.
* Reuse the loaded model; never load it per request.
* Keep API, tokenizer, LiteRT execution, configuration, and schemas separated.
* Public API should remain model-agnostic: clients send text and receive embeddings.
* Preserve the known model interface: input `[1,2048] int32`, output `[1,768] float32`.
* Support single and batch embedding requests.
* Default server binding: `127.0.0.1`.
* Write tests for core behavior and API behavior.
* Do not add dependencies unless necessary.
* Do not introduce abstractions without a concrete need.
* Prefer simple, readable implementations over over-engineering.

## Development

Before coding:

1. Read relevant parts of `SPEC.md`.
2. Inspect existing code.
3. Make the smallest change required.
4. Run relevant tests/checks.

Do not rewrite working code unnecessarily.

## Token Efficiency

* Do not explain obvious code.
* Keep responses concise.
* Do not repeat `SPEC.md`.
* When implementing, inspect only files relevant to the task.
* Make changes directly rather than proposing large rewrites.
* Report only: what changed, tests run, and any issue/blocker.
