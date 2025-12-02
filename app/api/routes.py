from __future__ import annotations

import logging
from fastapi import APIRouter, Depends

from app.models.api_requests import ArchitectureReviewRequest
from app.orchestrator.review_orchestrator import ReviewOrchestrator
# from app.orchestrator.orchestrator import review_workflow

from agent_framework import WorkflowOutputEvent

router = APIRouter(tags=["review"])
logger = logging.getLogger(__name__)


def get_orchestrator() -> ReviewOrchestrator:
    """
    Dependency factory for ReviewOrchestrator.

    FastAPI will call this function for any endpoint that declares
    `orchestrator: ReviewOrchestrator = Depends(get_orchestrator)`.

    Right now it just creates a new instance each time. If we later want a
    shared/singleton orchestrator or add configuration, we only need to
    change this function.
    """
    return ReviewOrchestrator()


@router.post("/review")
async def review_endpoint(
    request: ArchitectureReviewRequest,
    orchestrator: ReviewOrchestrator = Depends(get_orchestrator),
):

    has_metadata = isinstance(request.metadata, dict)
    metadata_keys = list(request.metadata.keys()) if has_metadata else None

    logger.info(
        f"Received /review request | has_metadata={has_metadata} | "
        f"metadata_keys={metadata_keys}"
    )

    result = await orchestrator.review(
        metadata=request.metadata,
    )

    return result

# @router.post("/review")
# async def review_endpoint(metadata: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Call the review workflow with the metadata JSON.
#     Collect the first WorkflowOutputEvent and return its data.
#     """
#     final_result: Dict[str, Any] | None = None

#     async for event in review_workflow.run_stream(metadata):
#         if isinstance(event, WorkflowOutputEvent):
#             final_result = event.data
#             # you can break here if you only expect one final output
#             break

#     # In practice you might want stricter checks here
#     return final_result or {
#         "status": "error",
#         "error_type": "no_output",
#         "message": "Workflow did not produce a final output.",
#     }
