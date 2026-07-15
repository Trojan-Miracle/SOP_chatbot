"""Document API endpoints for uploading and listing SOP documents."""

import os

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.session import Session
from app.schemas.document import DocumentResponse
from app.services.database import database_service
from app.services.document_service import SUPPORTED_EXTENSIONS, document_service

router = APIRouter()


@router.post("/upload", response_model=DocumentResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents_upload"][0])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_current_session),
):
    """Upload a SOP document (PDF, DOCX, TXT, or Markdown) for ingestion into the vector store.

    Args:
        request: The FastAPI request object for rate limiting.
        file: The uploaded document file.
        session: The current session from the auth token.

    Returns:
        DocumentResponse: The document record, initially in "processing" status.

    Raises:
        HTTPException: If the file type is unsupported or it exceeds the size limit.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not file.filename or ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422, detail=f"Unsupported file type. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    try:
        document_id = await document_service.upload(file.filename, content)
        document = await database_service.get_document(document_id)
        logger.info("document_upload_accepted", document_id=document_id, filename=file.filename, user_id=session.user_id)
        return DocumentResponse(
            document_id=document.id,
            filename=document.filename,
            status=document.status,
            chunk_count=document.chunk_count,
            error=document.error,
            created_at=document.created_at,
        )
    except Exception as e:
        logger.exception("document_upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=list[DocumentResponse])
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["documents_list"][0])
async def list_documents(request: Request, session: Session = Depends(get_current_session)):
    """List all uploaded SOP documents and their ingestion status.

    Args:
        request: The FastAPI request object for rate limiting.
        session: The current session from the auth token.

    Returns:
        list[DocumentResponse]: All document records, most recent first.
    """
    documents = await database_service.get_documents()
    return [
        DocumentResponse(
            document_id=d.id,
            filename=d.filename,
            status=d.status,
            chunk_count=d.chunk_count,
            error=d.error,
            created_at=d.created_at,
        )
        for d in documents
    ]
