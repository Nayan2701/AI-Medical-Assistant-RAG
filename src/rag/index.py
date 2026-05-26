from __future__ import annotations

import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def _faiss_files_exist(output_dir: Path, index_name: str) -> bool:
    # FAISS typically produces <name>.faiss and <name>.pkl
    return (output_dir / f"{index_name}.faiss").exists()


def get_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


def build_or_load_vectorstore(
    *,
    chunks,
    output_dir: Path,
    index_name: str,
    embedding_model_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embeddings(embedding_model_name)

    if not _faiss_files_exist(output_dir, index_name):
        vs = FAISS.from_documents(chunks, emb)
        vs.save_local(str(output_dir), index_name=index_name)
    # NOTE: allow_dangerous_deserialization is often required with FAISS local load.
    # Keep it, but only load indexes you created yourself.
    vs = FAISS.load_local(
        str(output_dir),
        emb,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )
    return vs