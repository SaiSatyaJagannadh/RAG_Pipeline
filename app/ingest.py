from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy

DATA_DIR = os.getenv("DATA_DIR", "data")

def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            #loaded_docs = []
            if ext == ".md":
                for d in UnstructuredMarkdownLoader(path).load():
                    docs.append(d)
            elif ext == ".pdf":
                for d in PyMuPDFLoader(path).load():
                    docs.append(d)
            elif ext == ".docx":
                for d in UnstructuredWordDocumentLoader(path).load():
                    docs.append(d)
            elif ext == ".txt":
                for d in TextLoader(path).load():
                    docs.append(d)
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()#list

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



async def run_ingest_async() -> dict:
    docs = _load_docs()
    chunks = _chunk(docs)
    store = await get_vector_store()
    await store.aadd_documents(chunks)
    print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks")
    #await _create_index(store)

    return {"documents": len(docs),"chunks":len(chunks)}

