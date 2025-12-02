# from __future__ import annotations

# from typing import Any, Dict

# from typing_extensions import Never
# from agent_framework import (
#     WorkflowBuilder,
#     WorkflowContext,
#     WorkflowOutputEvent,
#     executor,
# )

# from app.orchestrator.review_orchestrator import ReviewOrchestrator


# @executor(id="review_entry")
# async def review_entry(metadata: Dict[str, Any], ctx: WorkflowContext[Never, Dict[str, Any]]) -> None:
#     orchestrator = ReviewOrchestrator()
#     result = await orchestrator.review(metadata)
#     await ctx.yield_output(result)


# # Build the workflow
# review_workflow = (
#     WorkflowBuilder()
#     .set_start_executor(review_entry)
#     .build()
# )
