from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

from .uploads import DATA_DIR, SUPPORTED_EXTS

# No unstructured loaders: they pull a spacy model at runtime, which fails on
# read-only deploys (Streamlit Cloud). Chunking is done by the text splitter anyway,
# so plain text extraction is all these need to provide.
LOADERS = {
    ".md": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}
assert LOADERS.keys() == SUPPORTED_EXTS, "LOADERS and uploads.SUPPORTED_EXTS drifted"

def load_file(path: str, category: str) -> List[Document]:
    """Load a single file. Returns [] for unsupported extensions."""
    loader = LOADERS.get(os.path.splitext(path)[1].lower())
    if loader is None:
        return []
    docs = loader(path).load()
    for d in docs:
        d.metadata["category"] = category
    return docs

def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue
        relative_path = os.path.relpath(path,base)
        category = relative_path.split(os.sep)[0] if os.sep in relative_path else "general"

        try:
            docs.extend(load_file(path, category))
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs


def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 120
    )
    try:
        return splitter.split_documents(docs)
    except Exception:
        print(f"INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise
    
async def _create_index(store):
    index = HNSWIndex(
        name="hnsw_idx",
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        m=16,
        ef_construction=64
    )
    await store.aapply_vector_index(index,concurrently=True)
    print("Index Created Succesfully")


async def run_ingest_async() -> dict:
    docs = _load_docs()
    chunks = _chunk(docs)
    store = await get_vector_store()
    await store.aadd_documents(chunks)
    print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks")
    await _create_index(store)

    return {"documents": len(docs),"chunks":len(chunks)}


async def ingest_file_async(path: str, category: str) -> dict:
    """Ingest one already-saved file. HNSW index built by run_ingest_async covers new rows."""
    docs = load_file(path, category)
    if not docs:
        raise ValueError(f"Unsupported file type: {path}")
    chunks = _chunk(docs)
    store = await get_vector_store()
    await store.aadd_documents(chunks)
    print(f"INGEST: {path} -> {len(docs)} docs, {len(chunks)} chunks (category={category})")
    return {"documents": len(docs), "chunks": len(chunks)}

