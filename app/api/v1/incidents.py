"""Authenticated incident workflow using real knowledge-base retrieval and LLM."""

import os

from fastapi import Depends

from app.api.v1.auth import get_current_session
from app.core.incidents.live import LivePlanner
from app.core.incidents.routes import incident_router
from app.core.incidents.service import IncidentService
from app.core.incidents.store import IncidentStore
from app.core.limiter import limiter
from app.models.session import Session

store = IncidentStore(os.getenv("SOP_INCIDENT_DB", "data/incidents.sqlite3"))


async def incident_owner(session: Session = Depends(get_current_session)) -> str:
    """Derive ownership from the verified session, never from request input."""
    return str(session.user_id)


async def incident_service(owner: str = Depends(incident_owner)) -> IncidentService:
    """Inject the authenticated planner and async local store."""
    await store.initialize()
    return IncidentService(store, LivePlanner(owner), "live")


router = incident_router(incident_service, incident_owner, limiter)
