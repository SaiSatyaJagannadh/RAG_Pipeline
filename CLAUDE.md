# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full stack (postgres+pgvector, redis-stack, app on :8000)
docker compose up --build

# API only, from repo root
uvicorn app.api:app --reload --port 8000

# Ingest documents from data/ into pgvector
curl -X POST http://localhost:8000/ingest
curl http://localhost:8000/ingest/status

# Upload + ingest a single doc (pdf/docx/md/txt)
curl -X POST http://localhost:8000/upload -F file=@doc.pdf -F category=policies

# Upload path sanitization check (no project deps needed)
python test_upload.py

# Ask
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"How many PTO days carry forward?","category":"policies"}'

# Streamlit agent — must run from app/ (README says so; MCP server must be up first)
cd app && python -m streamlit run policy_agent.py

# RAGAS eval — also from app/ (hardcoded ../seed/qna_test.json, hits localhost:8000/ask)
cd app && python eval_ragas.py
```

No linter or CI. The only test is `test_upload.py`. `seed/qna_test.json` is **JSONL** despite the `.json` extension.

## Required env (`.env`, gitignored, read by docker-compose `env_file`)

`OPENAI_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL` (postgresql+psycopg async DSN), `REDIS_URL`.
Optional: `DATA_DIR` (default `data`), `RETRIEVAL_K` (default `5`).

## Architecture

Single Starlette app that **is** the MCP server — `app/api.py` does `app = mcp.streamable_http_app()` then bolts REST routes onto that same router. So `/mcp` (streamable HTTP), `/ask`, `/ingest`, `/` (static UI) all share one ASGI app. FastAPI is in requirements but unused; don't add FastAPI routers, use `app.router.add_route`.

Two consumers of the same backend:
- **HTTP**: `/ask` → `rag.answer_with_docs_async` → returns `{answer, sources, contexts}`.
- **MCP**: tools `rag_ask` (same function), `approve`, `reject` (stubs that just print). `policy_agent.py` is a Streamlit LangGraph agent that connects over `streamable_http` to `/mcp` and is told by its system prompt to call `rag_ask` then `approve`/`reject`.

### Category is the central abstraction

`ingest._load_docs` derives `category` from the **top-level folder name under `data/`** (`policies`, `faqs`, `guides`, `announcements`, `handbooks`, …) and writes it to `d.metadata["category"]`. That becomes a real Postgres **column**, not JSON — `utils.get_vector_store` declares `metadata_columns=["category"]` against `langchain_pg_embedding`, and `rag._build_chain` filters retrieval with `{"category": category}`. Adding a data folder adds a category with no code change; renaming one silently breaks any caller passing the old string (`/ask` defaults to `"policies"`, `policy_agent.py`'s prompt hardcodes `"policies"`).

The table is created by `init-db/init.sql`, which only runs on **first** postgres volume init. Changing the schema or embedding dimension (1536, from `text-embedding-3-small`) requires `docker compose down -v`.

### Retrieval chain (`app/rag.py`)

pgvector top-`RETRIEVAL_K` (async `AsyncPGVectorStore`) → Cohere `rerank-multilingual-v3.0` top-3 via `ContextualCompressionRetriever` → `gpt-4o-mini` stuff-documents chain. A global `RedisSemanticCache` (`distance_threshold=0.98`) is installed at import time via `set_llm_cache`, so importing `rag` requires Redis to be reachable.

### Ingest (`app/ingest.py`)

Loader picked by extension via the `LOADERS` map, whose keys must match `uploads.SUPPORTED_EXTS` (module-level assert). All loaders are deliberately pure-Python (`PyMuPDFLoader`, `Docx2txtLoader`, `TextLoader`) — do not reintroduce `unstructured`, which downloads a spaCy model at runtime and dies on Streamlit Cloud's read-only site-packages, and needs apt packages that conflict in its image. `load_file` handles one file and has three callers: the full `data/` sweep (`run_ingest_async`), the single-file upload path (`ingest_file_async`), and `policy_agent.extract_claims_from_document` (which uses it purely for text extraction, then has the LLM emit the `{employee, claims}` JSON so a PDF/Word claim doc feeds `process_claims` unchanged). Unsupported extensions are silently skipped — several files in `data/Course_files/` (`.pptx`, `.key`) never get ingested. Per-file failures are caught and logged, chunking failures abort. Chunks 900/120. Reruns **append** — there is no dedupe or delete, so ingesting twice duplicates every chunk. An HNSW cosine index is (re)applied after each run.

`POST /upload` (multipart `file` + `category`) writes into `data/<category>/` and ingests that one file synchronously. `uploads.safe_dest` is the trust boundary: it strips directory components from the filename, rejects dotfiles and unsupported extensions, and constrains `category` to `[A-Za-z0-9_-]{1,40}` — it lives in a dependency-free module so `test_upload.py` runs without the LLM stack installed. Uploads land in a bind-mounted `data/`, so they survive container restarts and are picked up by later full ingests (which will then duplicate them).

The full `data/` ingest runs as a single background `asyncio.Task` guarded by `_ingest_lock` + `_ingest_last` module state in `api.py`; a second POST while running returns 409. State is in-process, so it resets on reload/restart.

## Gotchas

- Imports use `langchain_classic.*` (LangChain 1.x layout) while `requirements.txt` pins nothing — a v0 `langchain` install will fail at import.
- docker-compose healthchecks `/health`, which does not exist; the app only serves `/mcp/health`. The container will report unhealthy but still work.
- `requirements.txt` has no version pins anywhere.
