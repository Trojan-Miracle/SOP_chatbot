"""Schemas for the tool-calling harness endpoint."""

from typing import (
    Any,
    List,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.schemas.base import BaseResponse
from app.schemas.chat import Message


class HarnessChatRequest(BaseModel):
    """Request model for the harness chat endpoint."""

    messages: List[Message] = Field(..., description="Conversation messages", min_length=1)


class ToolCallTrace(BaseModel):
    """One tool invocation the model made while answering."""

    tool: str = Field(description="Name of the tool that was called")
    args: dict[str, Any] = Field(description="Arguments the model passed to the tool")
    result: str = Field(description="The tool's return value")


class HarnessChatResponse(BaseResponse):
    """Response model for the harness chat endpoint."""

    messages: List[Message] = Field(..., description="Conversation messages, including the new answer")
    tool_calls: List[ToolCallTrace] = Field(default_factory=list, description="Tools invoked while answering")
