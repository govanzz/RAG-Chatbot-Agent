"""Document chunking, embedding, FAISS storage, and retrieval."""

from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    BASE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    MIN_RELEVANCE_SCORE,
    RAW_PAGES_DIR,
    VECTORSTORE_DIR,
)
from utils import clean_inline_text, clean_multiline_text, read_json, utc_now_iso, write_json


class VectorStoreMissingError(FileNotFoundError):
    """Raised when the FAISS index or metadata file has not been created yet."""


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return faiss


def _require_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer


def _load_embedding_model(local_files_only: bool = False):
    SentenceTransformer = _require_sentence_transformer()
    try:
        if local_files_only:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                return SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    local_files_only=True,
                )
        return SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=False,
        )
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                "The embedding model is not available from the local cache. "
                "Run `python ingest_web.py` once with internet access so "
                f"{EMBEDDING_MODEL_NAME} can be downloaded."
            ) from exc
        raise


@dataclass
class RawPage:
    path: Path
    title: str
    source_url: str
    content: str


def vectorstore_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


def get_indexed_chunk_count() -> int:
    metadata = read_json(METADATA_PATH, default={})
    if isinstance(metadata, dict):
        return len(metadata.get("chunks", []))
    if isinstance(metadata, list):
        return len(metadata)
    return 0


def _parse_raw_page(path: Path) -> RawPage:
    raw = path.read_text(encoding="utf-8")
    header, content = raw.split("\n---\n", 1) if "\n---\n" in raw else ("", raw)

    title = path.stem
    source_url = ""
    for line in header.splitlines():
        if line.startswith("Page Title:"):
            title = clean_inline_text(line.split(":", 1)[1])
        elif line.startswith("Source URL:"):
            source_url = clean_inline_text(line.split(":", 1)[1])

    return RawPage(
        path=path,
        title=title or path.stem,
        source_url=source_url,
        content=clean_multiline_text(content),
    )


def _split_sections(content: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    for line in content.splitlines():
        heading_match = re.match(r"^#{1,4}\s+(.+)$", line)
        if heading_match:
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                buffer = []
            heading = clean_inline_text(heading_match.group(1))
            buffer.append(line)
        else:
            buffer.append(line)

    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))

    return [(section_heading, text) for section_heading, text in sections if text]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_start = start + max(1, chunk_size // 2)
            newline_boundary = text.rfind("\n", search_start, end)
            sentence_boundary = text.rfind(". ", search_start, end)
            boundary = max(newline_boundary, sentence_boundary)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def load_raw_pages(raw_dir: Path = RAW_PAGES_DIR) -> list[RawPage]:
    return [_parse_raw_page(path) for path in sorted(raw_dir.glob("*.txt"))]


def build_chunks(raw_dir: Path = RAW_PAGES_DIR) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in load_raw_pages(raw_dir):
        relative_path = str(page.path.relative_to(BASE_DIR))
        for section_heading, section_text in _split_sections(page.content):
            for chunk in chunk_text(section_text):
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "source_url": page.source_url,
                        "page_title": page.title,
                        "section_heading": section_heading,
                        "raw_page_path": relative_path,
                        "text": chunk,
                    }
                )
    return chunks


def build_vectorstore(raw_dir: Path = RAW_PAGES_DIR) -> int:
    chunks = build_chunks(raw_dir)
    if not chunks:
        raise RuntimeError(
            f"No scraped .txt pages found in {raw_dir}. Run: python ingest_web.py"
        )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss = _require_faiss()

    model = _load_embedding_model(local_files_only=False)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    write_json(
        METADATA_PATH,
        {
            "created_at": utc_now_iso(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks": chunks,
        },
    )
    return len(chunks)


def _load_metadata_chunks() -> list[dict[str, Any]]:
    metadata = read_json(METADATA_PATH, default={})
    if isinstance(metadata, dict):
        return metadata.get("chunks", [])
    if isinstance(metadata, list):
        return metadata
    return []


class MandaiRetriever:
    def __init__(self) -> None:
        if not vectorstore_exists():
            raise VectorStoreMissingError("Vectorstore is missing. Run: python ingest_web.py")

        faiss = _require_faiss()

        self.metadata = _load_metadata_chunks()
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.model = _load_embedding_model(local_files_only=True)

    @property
    def indexed_chunk_count(self) -> int:
        return len(self.metadata)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not self.metadata:
            return []

        top_k = max(1, min(top_k, len(self.metadata)))
        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict[str, Any]] = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue
            chunk = dict(self.metadata[int(index_id)])
            chunk["score"] = float(score)
            results.append(chunk)
        return results


def has_sufficient_context(chunks: list[dict[str, Any]]) -> bool:
    if not chunks:
        return False
    best_score = max(float(chunk.get("score", 0.0)) for chunk in chunks)
    return best_score >= MIN_RELEVANCE_SCORE


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        section = chunk.get("section_heading") or "General"
        blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"Page title: {chunk.get('page_title', 'Untitled')}",
                    f"URL: {chunk.get('source_url', '')}",
                    f"Section: {section}",
                    "Content:",
                    clean_multiline_text(chunk.get("text", "")),
                ]
            )
        )
    return "\n\n".join(blocks)


def unique_sources(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for chunk in chunks:
        url = chunk.get("source_url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "title": chunk.get("page_title") or "Mandai page",
                "url": url,
            }
        )
    return sources
