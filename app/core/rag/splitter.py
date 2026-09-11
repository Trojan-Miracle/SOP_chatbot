"""Chunking utilities for SOP document text."""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import settings

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.RAG_CHUNK_SIZE,
    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
)

_MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MARKDOWN_HEADERS, strip_headers=False)


def split_pages(pages: list[tuple[int, str]], *, document_id: str, filename: str) -> list[Document]:
    """Split extracted pages into overlapping, fixed-size chunks with source metadata.

    Used for formats without native structure to split on (PDF, DOCX, plain text).

    Args:
        pages: List of (page_number, text) tuples from a ``load_*_pages`` function.
        document_id: The persisted document's ID, stored as chunk metadata.
        filename: The original filename, stored as chunk metadata for citations.

    Returns:
        List of LangChain ``Document`` chunks ready to embed.
    """
    chunks: list[Document] = []
    for page_number, text in pages:
        for i, chunk_text in enumerate(_splitter.split_text(text)):
            chunk_id = f"{document_id}:{page_number}:{i}"
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                    },
                )
            )
    return chunks


def split_markdown(text: str, *, document_id: str, filename: str) -> list[Document]:
    """Split Markdown text along its own header structure instead of a fixed size.

    Each section under a heading becomes its own chunk (further split by
    ``RecursiveCharacterTextSplitter`` only if a section is still too long),
    so a step under "## Cleaning Procedure" doesn't get arbitrarily cut in
    half the way fixed-size chunking would.

    Args:
        text: Full Markdown source.
        document_id: The persisted document's ID, stored as chunk metadata.
        filename: The original filename, stored as chunk metadata for citations.

    Returns:
        List of LangChain ``Document`` chunks ready to embed.
    """
    sections = _markdown_header_splitter.split_text(text)
    chunks: list[Document] = []
    for section_idx, section in enumerate(sections):
        heading = " > ".join(
            section.metadata.get(level, "") for _, level in _MARKDOWN_HEADERS if level in section.metadata
        )
        sub_chunks = _splitter.split_text(section.page_content) or [section.page_content]
        for i, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{document_id}:{section_idx}:{i}"
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": filename,
                        "page": 1,
                        "section": heading or "(no heading)",
                    },
                )
            )
    return chunks
