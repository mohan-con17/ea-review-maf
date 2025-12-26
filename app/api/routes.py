from __future__ import annotations

import logging
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from starlette.concurrency import run_in_threadpool
from app.config.settings import settings
from typing import Optional

from app.models.api_requests import ArchitectureReviewRequest
from app.orchestrator.review_orchestrator import ReviewOrchestrator
from app.services.blob_service import BlobLogService

router = APIRouter(tags=["review"])
logger = logging.getLogger(__name__)
orchestrator = ReviewOrchestrator()

def sse_event(event: str, data: dict) -> str:
    """
    Format an SSE event.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@router.post("/review/stream")
async def review_stream(request: Request, body: dict):
    """
    SSE endpoint: streams progress events + final response.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def progress_cb(stage: str, status: str, payload: dict):
        # You can pick your own event names; here we use "stage"
        event_payload = {
            "stage": stage,
            "status": status,
            "payload": payload,
        }
        await queue.put(sse_event("stage", event_payload))

    async def run_review():
        try:
            # Call orchestrator with progress callback
            result = await orchestrator.review(body.get("metadata", {}), progress_cb)
            # Final event
            await queue.put(sse_event("final", result))
        except Exception as e:
            await queue.put(
                sse_event("error", {"message": str(e)})
            )
        finally:
            # Sentinel to stop generator
            await queue.put(None)

    asyncio.create_task(run_review())

    async def event_generator():
        try:
            while True:
                # client disconnected?
                if await request.is_disconnected():
                    break
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

# -----------------------------------------------------------
# Dependency for orchestrator
# -----------------------------------------------------------
def get_orchestrator() -> ReviewOrchestrator:
    return ReviewOrchestrator()


# -----------------------------------------------------------
# POST /review   (EXISTING API)
# -----------------------------------------------------------
# @router.post("/review")
# async def review_endpoint(
#     request: ArchitectureReviewRequest,
#     orchestrator: ReviewOrchestrator = Depends(get_orchestrator),
# ):
#     try:
#         result = await orchestrator.review(metadata=request.metadata)
#         return result
#     except Exception as e:
#         logger.error(f"Review orchestration failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/review")
async def review_endpoint(request: Request, body: dict):
    """
    Unified REVIEW endpoint:
    - Streams live orchestrator stages (SSE)
    - Sends final response on completion
    """

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def progress_cb(stage: str, status: str, payload: dict):
        event_payload = {
            "stage": stage,
            "status": status,
            "payload": payload,
        }
        await queue.put(sse_event("stage", event_payload))

    async def run_review():
        try:
            result = await orchestrator.review(
                body.get("metadata", {}),
                progress_cb=progress_cb,
            )
            await queue.put(sse_event("final", result))
        except Exception as e:
            await queue.put(sse_event("error", {"message": str(e)}))
        finally:
            await queue.put(None)

    asyncio.create_task(run_review())

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

_blob_service_instance: Optional[BlobLogService] = None

def get_blob_service() -> BlobLogService:
    global _blob_service_instance
    if _blob_service_instance is None:
        _blob_service_instance = BlobLogService()
    return _blob_service_instance

# ---------------------------
# Logs endpoints (the 4 requested APIs)
# ---------------------------

@router.get("/logs/all-sessions")
async def api_get_all_sessions(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=1000)):
    svc = get_blob_service()
    try:
        result = await run_in_threadpool(svc.list_all_sessions, page, page_size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/dates")
async def api_get_all_dates():
    svc = get_blob_service()
    try:
        dates = await run_in_threadpool(svc.list_all_dates)
        return {"dates": dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/months")
async def api_get_months_by_year():
    svc = get_blob_service()
    try:
        months = await run_in_threadpool(svc.list_months_by_year)
        return {"months_by_year": months}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/session")
async def api_get_session(
    session_id: str = Query(..., description="Session ID (filename without .json)"),
    month: Optional[str] = Query(None, description="3-letter month abbreviation, e.g. Dec"),
    year: Optional[str] = Query(None, description="4-digit year, e.g. 2025"),
    date: Optional[str] = Query(None, description="Date in DD-MM-YYYY"),
):
    svc = get_blob_service()
    # If month and year are present but sent separately, build month param as "Mon YYYY"
    if month and year:
        month_param = f"{month} {year}"
    else:
        month_param = None

    try:
        details = await run_in_threadpool(svc.get_session_details, session_id, month, year, date)
        if details is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
