"""Document ingestion service: parse -> chunk -> embed -> store SOP documents."""

import asyncio
import os
import uuid

from app.core.config import settings
from app.core.logging import logger
from app.core.rag.bm25_index import get_bm25_index
from app.core.rag.loader import load_docx_pages, load_pdf_pages, load_text_pages
from app.core.rag.splitter import split_markdown, split_pages
from app.core.rag.vectorstore import get_vectorstore
from app.models.document import DocumentStatus
from app.services.database import database_service
from app.utils import spawn_background_task

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentService:
    """Orchestrates SOP document ingestion into the vector store.

    Upload persists the file and a ``processing`` document record
    synchronously, then hands the parse/chunk/embed work off to a background
    task so the HTTP request returns immediately.
    """

    async def upload(self, filename: str, content: bytes) -> str:
        """Save an uploaded document and kick off background ingestion.

        Args:
            filename: Original uploaded filename.
            content: Raw file bytes.

        Returns:
            The generated document ID, immediately usable to poll status.
        """
        document_id = str(uuid.uuid4())
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}_{filename}")
        await asyncio.to_thread(self._write_file, file_path, content)

        await database_service.create_document(document_id, filename, file_path)
        spawn_background_task(self._ingest(document_id, filename, file_path))
        return document_id

    @staticmethod
    def _write_file(file_path: str, content: bytes) -> None:
        with open(file_path, "wb") as f:
            f.write(content)

    async def _ingest(self, document_id: str, filename: str, file_path: str) -> None:
        """Parse, chunk, embed, and store a document; update its status when done."""
        try:
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".md":
                pages = await asyncio.to_thread(load_text_pages, file_path)
                text = pages[0][1] if pages else ""
                if not text.strip():
                    raise ValueError("no extractable text found in file")
                chunks = split_markdown(text, document_id=document_id, filename=filename)
            else:
                if ext == ".pdf":
                    pages = await asyncio.to_thread(load_pdf_pages, file_path)
                elif ext == ".docx":
                    pages = await asyncio.to_thread(load_docx_pages, file_path)
                elif ext == ".txt":
                    pages = await asyncio.to_thread(load_text_pages, file_path)
                else:
                    raise ValueError(f"unsupported file type '{ext}' (supported: {sorted(SUPPORTED_EXTENSIONS)})")

                if not pages:
                    raise ValueError("no extractable text found (scanned/image-only files are not supported)")
                chunks = split_pages(pages, document_id=document_id, filename=filename)

            vectorstore = get_vectorstore()
            chunk_ids = [c.metadata["chunk_id"] for c in chunks]
            await vectorstore.aadd_documents(chunks, ids=chunk_ids)

            # BM25 index has no incremental-update API in this implementation —
            # cheap enough at demo scale to just rebuild from the full collection.
            await asyncio.to_thread(get_bm25_index().rebuild)

            await database_service.update_document_status(document_id, DocumentStatus.READY, chunk_count=len(chunks))
            logger.info("document_ingested", document_id=document_id, filename=filename, chunk_count=len(chunks))
        except Exception as e:
            logger.exception("document_ingestion_failed", document_id=document_id, filename=filename, error=str(e))
            await database_service.update_document_status(document_id, DocumentStatus.FAILED, error=str(e))


document_service = DocumentService()
