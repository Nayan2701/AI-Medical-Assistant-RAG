from __future__ import annotations

from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from .prompt import RAG_PROMPT


def make_llm(*, model: str, google_api_key: str | None, temperature: float = 0):
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env or environment.")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key,
    )


def retrieve(retriever, query: str, k: int) -> List[Dict[str, Any]]:
    # LangChain deprecation note:
    # get_relevant_documents is being phased in favor of retriever.invoke(query).
    # We'll try invoke first, fallback for compatibility.
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)

    out = []
    for i, d in enumerate(docs[:k], 1):
        out.append(
            {
                "rank": i,
                "content": (d.page_content or "").strip(),
                "source": d.metadata.get("source") or d.metadata.get("path") or "unknown",
                "page": d.metadata.get("page") or d.metadata.get("page_number"),
            }
        )
    return out


def build_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    pieces = []
    used = 0
    for i, ch in enumerate(chunks, 1):
        tag = f"[S{i}]"
        text = ch["content"].replace("\n", " ").strip()
        block = f"{tag} {text}"
        if used + len(block) > max_chars:
            break
        pieces.append(block)
        used += len(block)
    return "\n\n".join(pieces)


def build_citation_block(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        page = f", p.{ch['page']}" if ch.get("page") is not None else ""
        lines.append(f"[S{i}] {ch['source']}{page}")
    return "\n".join(lines) if lines else "No sources."


def generate_answer(llm, question: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "The provided context does not contain that information."

    context = build_context(chunks)
    citations = build_citation_block(chunks)

    msg = RAG_PROMPT.format_messages(
        question=question,
        context=context,
        citations=citations,
    )

    # Avoid deprecated direct-call patterns:
    # Prefer invoke; fallback to callable if needed.
    try:
        resp = llm.invoke(msg)
    except Exception:
        resp = llm(msg)

    return resp.content.strip()