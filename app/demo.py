"""Local offline portfolio demo; intentionally separate from authenticated live app."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.incidents.demo import DemoPlanner
from app.core.incidents.routes import incident_router
from app.core.incidents.service import IncidentService
from app.core.incidents.store import IncidentStore

store = IncidentStore(os.getenv("SOP_DEMO_DB", "data/demo-incidents.sqlite3"))
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize durable local demo storage."""
    await store.initialize()
    yield


async def demo_owner() -> str:
    """Use one local identity; this app is for loopback demonstrations only."""
    return "local-demo"


async def demo_service() -> IncidentService:
    """Provide deterministic fixtures with the same workflow and persistence."""
    return IncidentService(store, DemoPlanner(), "demo")


app = FastAPI(title="SOP Copilot · Offline Demo", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]
app.include_router(incident_router(demo_service, demo_owner, limiter), prefix="/api/v1/incidents")


@app.get("/")
@limiter.limit("60/minute")
async def index(request: Request) -> FileResponse:
    """Serve the standalone incident workbench."""
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "incidents.html")
