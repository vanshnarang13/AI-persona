from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class Document:
    content: str
    source: str                        # e.g. "corpus/projects/proj_atlasrag"
    source_type: str                   # "corpus" | "resume"
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None
    # metadata keys: section_title, file_path, char_count


def _derive_source(path: Path, corpus_root: Path) -> str:
    """Convert absolute path to a relative source string like corpus/projects/proj_atlasrag"""
    try:
        rel = path.relative_to(corpus_root)
    except ValueError:
        rel = path
    # strip extension
    return str(rel.with_suffix(""))


def partition_file(path: Path, corpus_root: Path, source_type: str = "corpus") -> list[Document]:
    """
    Split a markdown file into Documents by ## headers.
    Each ## section becomes one Document. Content before the first ## is its own Document.
    """
    text = path.read_text(encoding="utf-8")
    source = _derive_source(path, corpus_root)

    # Split on ## headers (keep the header in the section)
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    docs: list[Document] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section title from first line if it starts with ##
        lines = section.splitlines()
        if lines[0].startswith("## "):
            section_title = lines[0].lstrip("#").strip()
        elif lines[0].startswith("# "):
            section_title = lines[0].lstrip("#").strip()
        else:
            section_title = "overview"

        # Prepend source anchor so BM25 can match on project/person name
        anchored_content = f"{source} > {section_title}:\n{section}"

        docs.append(Document(
            content=anchored_content,
            source=source,
            source_type=source_type,
            metadata={
                "section_title": section_title,
                "file_path": str(path),
                "char_count": len(anchored_content),
            },
        ))

    return docs


def partition_corpus(corpus_dir: Path, source_type: str = "corpus") -> list[Document]:
    """Walk all .md files under corpus_dir and partition each."""
    docs: list[Document] = []
    for md_file in sorted(corpus_dir.rglob("*.md")):
        docs.extend(partition_file(md_file, corpus_dir, source_type))
    return docs
