# RAG_Pipeline

# To run the agents 
python -m streamlit run policy_agent.py

# working app link

https://employee-expenses-tracking-agent.streamlit.app/

# Streamlit Cloud deploy

Main file path: `app/policy_agent.py`

No `packages.txt` — the loaders are pure-Python wheels (pymupdf for pdf, docx2txt for docx).
Adding `libmagic1` breaks the build: Streamlit's image mixes Debian bullseye and trixie repos
and the versions conflict. Don't add `unstructured` back either — it downloads a spaCy model
at runtime into a read-only site-packages.

Secrets (Settings -> Secrets), same keys as `.env`:

```toml
OPENAI_API_KEY = "..."
COHERE_API_KEY = "..."
DATABASE_URL = "postgresql+psycopg://user:pass@host:5432/db"   # must be reachable from Streamlit Cloud
REDIS_URL = "redis://..."
```

The sidebar "Knowledge base" uploader takes pdf/docx/md/txt and ingests in-process, so it
works without the API server running. The claims agent still needs the MCP server
(`uvicorn app.api:app`) reachable at the MCP Server URL in the sidebar.

Uploaded files are written to `data/<category>/` — on Streamlit Cloud that disk is
ephemeral, but the embeddings persist in Postgres.
