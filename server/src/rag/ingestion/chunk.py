from __future__ import annotations
import tiktoken
from .partition import Document

_ENCODER = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 400
OVERLAP_TOKENS = 50


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_by_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += max_tokens - overlap

    return chunks


def chunk_document(
    doc: Document,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[Document]:
    """
    If a Document fits within max_tokens, return it as-is.
    Otherwise split with a sliding token window, preserving the source anchor
    on every chunk so BM25 can still find it by project/section name.
    """
    if _count_tokens(doc.content) <= max_tokens:
        return [doc]

    # Extract the anchor prefix (first line: "source > section_title:")
    lines = doc.content.splitlines()
    anchor = lines[0] if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else doc.content

    text_chunks = _split_by_tokens(body, max_tokens - _count_tokens(anchor + "\n"), overlap)

    result: list[Document] = []
    for i, chunk_text in enumerate(text_chunks):
        content = f"{anchor}\n{chunk_text}" if anchor else chunk_text
        meta = {**doc.metadata, "chunk_index": i, "total_chunks": len(text_chunks)}
        result.append(Document(
            content=content,
            source=doc.source,
            source_type=doc.source_type,
            metadata=meta,
        ))

    return result


def chunk_documents(docs: list[Document], **kwargs) -> list[Document]:
    result: list[Document] = []
    for doc in docs:
        result.extend(chunk_document(doc, **kwargs))
    return result
