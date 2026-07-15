"""This file contains the document model for the application."""

from enum import Enum
from typing import Optional

from sqlmodel import Field

from app.models.base import BaseModel


class DocumentStatus(str, Enum):
    """Lifecycle status of an uploaded SOP document."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(BaseModel, table=True):
    """Document model for tracking uploaded SOP PDFs and their ingestion state.

    Attributes:
        id: The primary key (UUID string).
        filename: Original uploaded filename.
        file_path: Path to the stored PDF on disk.
        status: Ingestion lifecycle status.
        chunk_count: Number of vector chunks produced once ready.
        error: Error message when status is failed.
    """

    id: str = Field(primary_key=True)
    filename: str
    file_path: str
    status: DocumentStatus = Field(default=DocumentStatus.PROCESSING)
    chunk_count: int = Field(default=0)
    error: Optional[str] = Field(default=None)
