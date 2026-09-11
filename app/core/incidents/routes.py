"""Router factory keeps real authentication separate from the local demo identity."""

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter

from app.core.incidents.schemas import Decision, Incident, IncidentInput, Revision
from app.core.incidents.service import IncidentService
from app.core.incidents.store import ConflictError


def incident_router(service_dependency: Callable, owner_dependency: Callable, limiter: Limiter) -> APIRouter:
    """Build identical versioned endpoints for live and demo adapters."""
    router = APIRouter()

    @router.post("", response_model=Incident)
    @limiter.limit("20/minute")
    async def create(
        request: Request,
        facts: IncidentInput,
        owner: str = Depends(owner_dependency),
        service: IncidentService = Depends(service_dependency),
    ) -> Incident:
        """Start an incident and clarify missing facts before planning."""
        return await service.prepare(facts, owner)

    @router.get("/{incident_id}", response_model=Incident)
    @limiter.limit("60/minute")
    async def get(
        request: Request,
        incident_id: str,
        owner: str = Depends(owner_dependency),
        service: IncidentService = Depends(service_dependency),
    ) -> Incident:
        """Resume an owned incident after a refresh or process restart."""
        try:
            return await service.store.get(incident_id, owner)
        except KeyError:
            raise HTTPException(404, "事件不存在。")

    @router.put("/{incident_id}", response_model=Incident)
    @limiter.limit("20/minute")
    async def revise(
        request: Request,
        incident_id: str,
        revision: Revision,
        owner: str = Depends(owner_dependency),
        service: IncidentService = Depends(service_dependency),
    ) -> Incident:
        """Replace facts and invalidate the old proposal before another approval."""
        try:
            return await service.prepare(revision.facts, owner, incident_id, revision.expected_version)
        except KeyError:
            raise HTTPException(404, "事件不存在。")
        except ConflictError as exc:
            raise HTTPException(409, str(exc))

    @router.post("/{incident_id}/decision", response_model=Incident)
    @limiter.limit("20/minute")
    async def decide(
        request: Request,
        incident_id: str,
        decision: Decision,
        owner: str = Depends(owner_dependency),
        service: IncidentService = Depends(service_dependency),
    ) -> Incident:
        """Approve or reject the exact proposal version shown to the operator."""
        try:
            return await service.store.decide(incident_id, owner, decision)
        except KeyError:
            raise HTTPException(404, "事件不存在。")
        except ConflictError as exc:
            raise HTTPException(409, str(exc))

    return router
