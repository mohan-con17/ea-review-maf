from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext
from app.prompts.prompt_registry import PromptRegistry
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class FormattingAgent:
    """
    FormattingAgent
    ----------------
    - Final response formatter for FE
    - NEVER throws
    - LLM traces captured for scoring
    """

    def __init__(self) -> None:
        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )

        self._token_counter = TokenCounter()

        success_prompt = PromptRegistry.get("formatting_success", "v1")
        failure_prompt = PromptRegistry.get("formatting_failure", "v1")

        self._success_agent = ChatAgent(
            chat_client=chat_client,
            instructions=success_prompt["messages"]["system"],
            name="FormattingAgentSuccess",
            max_output_tokens=success_prompt["model"]["max_tokens"],
            temperature=success_prompt["model"]["temperature"],
        )

        self._failure_agent = ChatAgent(
            chat_client=chat_client,
            instructions=failure_prompt["messages"]["system"],
            name="FormattingAgentFailure",
            max_output_tokens=failure_prompt["model"]["max_tokens"],
            temperature=failure_prompt["model"]["temperature"],
        )

    # ==============================================================
    # FAILURE FORMATTER
    # ==============================================================

    async def format_failure_response(
        self,
        ctx: ReviewSessionContext,
        failure_details: List[Dict[str, Any]],
        failure_stage: str,
    ) -> Dict[str, Any]:
        """
        Formats a failure response.
        NEVER throws.
        """

        user_prompt = (
            f"Review ID: {ctx.review_id}\n"
            f"Failure Stage: {failure_stage}\n\n"
            f"Failure Details:\n"
            + json.dumps(failure_details, indent=2)
        )

        output_text = ""

        try:
            # ---------------- Token counting ----------------
            ctx.last_input_tokens = self._token_counter.count_text(user_prompt)

            response = await self._failure_agent.run(user_prompt)
            output_text = response.text or ""

            ctx.last_output_tokens = self._token_counter.count_text(output_text)
            ctx.last_total_tokens = (
                ctx.last_input_tokens + ctx.last_output_tokens
            )

            # ---------------- LLM TRACE (SUCCESS) ----------------
            ctx.llm_traces.append({
                "agent": "formatting",
                "prompt": user_prompt,
                "response": output_text,
                "status": "success",
            })

        except Exception as e:
            logger.exception("[%s] Failure formatter LLM failed", ctx.review_id)

            # ---------------- LLM TRACE (FAILURE) ----------------
            ctx.llm_traces.append({
                "agent": "formatting",
                "prompt": user_prompt,
                "response": "",
                "status": "failure",
            })

            output_text = (
                "The review failed due to an internal error. "
                "Detailed formatting could not be generated."
            )

        payload = {
            "review_id": ctx.review_id,
            "status": "failure",
            "stage": failure_stage,
            "summary": output_text,
            "issues": failure_details,
            "data": {},
        }

        ctx.formatting_summary = payload
        return payload

    # ==============================================================
    # SUCCESS FORMATTER
    # ==============================================================

    async def format_success_response(self, ctx: ReviewSessionContext) -> Dict[str, Any]:
        """
        Formats a success response.
        NEVER throws.
        """

        combined_payload = {
            "demographics": getattr(ctx, "demographics_from_json", None),
            "image_analysis": getattr(ctx, "image_analysis", None),
            "remediation": getattr(ctx, "remediation_result", None),
        }

        user_prompt = json.dumps(combined_payload, default=str, indent=2)
        output_text = ""

        try:
            # ---------------- Token counting ----------------
            ctx.last_input_tokens = self._token_counter.count_text(user_prompt)

            response = await self._success_agent.run(user_prompt)
            output_text = response.text or ""

            ctx.last_output_tokens = self._token_counter.count_text(output_text)
            ctx.last_total_tokens = (
                ctx.last_input_tokens + ctx.last_output_tokens
            )

            # ---------------- LLM TRACE (SUCCESS) ----------------
            ctx.llm_traces.append({
                "agent": "formatting",
                "prompt": user_prompt,
                "response": output_text,
                "status": "success",
            })

        except Exception as e:
            logger.exception("[%s] Success formatter LLM failed", ctx.review_id)

            # ---------------- LLM TRACE (FAILURE) ----------------
            ctx.llm_traces.append({
                "agent": "formatting",
                "prompt": user_prompt,
                "response": "",
                "status": "failure",
            })

            output_text = (
                "The review completed successfully, "
                "but a formatted summary could not be generated."
            )

        payload = {
            "review_id": ctx.review_id,
            "status": "success",
            "stage": "completed",
            "summary": output_text,
            "issues": [],
            "data": {},
        }

        ctx.formatting_summary = payload
        return payload
