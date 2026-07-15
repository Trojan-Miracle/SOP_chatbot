"""This file contains the document schemas for the application."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Response model for a single document's ingestion state."""

    document_id: str = Field(..., description="The document's unique ID")
    filename: str = Field(..., description="Original uploaded filename")
    status: DocumentStatus = Field(..., description="Ingestion lifecycle status")
    chunk_count: int = Field(..., description="Number of vector chunks produced once ready")
    error: str | None = Field(default=None, description="Error message when status is failed")
    created_at: datetime = Field(..., description="When the document was uploaded")
