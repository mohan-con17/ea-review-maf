from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext
from app.orchestrator.planner_kernel import PlanDecision

logger = logging.getLogger(__name__)

class FormattingAgent:

    def __init__(self) -> None:

        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )
        # instructions = (
        #     """
        #     You are the Formatting Agent in an enterprise architecture review workflow for a regulated BFSI environment. 

        #     Your task is to convert the JSON provided to you into concise, professional narratives. 
            
        #     If there are any score related details in the inputs, focus only on the similarity score. Do not include any other scores or raw metadata.

        #     Formatting requirements:
        #     - Use short paragraphs. 
        #     - Use numbered or bulleted lists when suitable.
        #     - Present the similarity score clearly in one short paragraph.
        #     - Use bulleted or numbered lists only if it improves readability.
        #     - Avoid repeating field names or technical keys; summarize meaningfully.
        #     - Avoid speculation or generic statements; base content strictly on the data.
        #     """
        # )
        
        self._instructions = """
        You are the Formatting Agent in an enterprise architecture review workflow for a regulated BFSI environment. 

        Your task is to convert the JSON or structured input provided to you into concise, professional narratives. 

        Guidelines:
        - Use short paragraphs.
        - Use numbered or bulleted lists when suitable.
        - If there are any score related details in the inputs, present similarity score details only in one short paragraph.
        - Avoid repeating field names or raw metadata; summarize meaningfully.
        - Avoid speculation; base content strictly on the provided data.
        - Tone: professional, concise, and actionable.
        """

        self._agent = ChatAgent(
            chat_client=chat_client,
            instructions=self._instructions,
            name="FormatterAgent",
            max_output_tokens=5120,
            temperature=0.5,
        )

    async def format_failure_response(
        self,
        ctx: ReviewSessionContext,
        failure_details: List[Dict[str, Any]],
        failure_stage: str,
    ) -> Dict[str, Any]:

        logger.info(
            f"[{ctx.review_id}] Formatting failure response",
        )
        if not failure_details:
            failure_details_text = "No specific details provided."
        else:
            # failure_details_text = "\n".join(
            #     f"- [{d.get('level', 'ERROR')}] {d.get('field', '')}: {d.get('message', str(d))}"
            #     for d in failure_details
            # )
            
            failure_details_text = "\n".join(
                f"- [{getattr(d, 'level', d.get('level', 'ERROR') if isinstance(d, dict) else 'ERROR')}] "
                f"{getattr(d, 'field', d.get('field', '') if isinstance(d, dict) else '')}: "
                f"{getattr(d, 'message', d.get('message', str(d)) if isinstance(d, dict) else str(d))}"
                for d in failure_details
            )

        user_prompt = (
            f"Review ID: {ctx.review_id}\n"
            f"Stage: {failure_stage}\n\n"
            "The review could not proceed due to the following issues:\n"
            f"{failure_details_text}\n\n"
            "Your task is to generate a clear, professional message to the user, "
            "including:\n"
            "1. A short introduction acknowledging the Review ID.\n"
            "2. A clear statement that the request cannot proceed.\n"
            "3. A structured list of the required corrections or issues.\n"
            "4. A polite closing encouraging resubmission after fixes."
        )

        response = await self._agent.run(user_prompt)
        message_text = response.text or ""

        return {
            "review_id": ctx.review_id,
            "status": "failure",
            "stage": failure_stage,
            "issues": failure_details,
            "message": message_text,
        }

    async def format_success_response(
        self,
        ctx: ReviewSessionContext,
        plan: PlanDecision,
    ) -> Dict[str, Any]:

        logger.info(
            "SummarizerAgent started in dynamic final-summary mode",
            extra={"review_id": ctx.review_id},
        )

        def to_dict(obj):
            if obj is None:
                return None
            if is_dataclass(obj):
                return asdict(obj)
            if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
                try:
                    return obj.dict()
                except Exception:
                    pass
            return getattr(obj, "__dict__", obj)

        # Flatten all agents from plan stages
        stages = getattr(plan, "stages", []) or []
        planned_agents = [agent for stage in stages for agent in stage if isinstance(agent, str)]

        agent_outputs = {}
        missing_agents = []

        for agent_name in planned_agents:
            # Try to fetch output dynamically from ctx using common patterns
            attr_candidates = [
                agent_name,
                f"{agent_name}_result",
                f"{agent_name}_results",
                f"{agent_name}_snapshot",
                f"{agent_name}_from_json",
            ]

            agent_data = None
            for attr in attr_candidates:
                agent_data = getattr(ctx, attr, None)
                if agent_data is not None:
                    break

            if agent_data is None:
                missing_agents.append(agent_name)

            agent_outputs[agent_name] = to_dict(agent_data)

        payload = {
            "review_id": ctx.review_id,
            "status": "success",
            "planner": to_dict(plan),
            "agents": agent_outputs,
            "missing_agents": missing_agents,
        }

        # If at least one agent produced output, generate LLM summary for the last agent
        last_agent_name = planned_agents[-1] if planned_agents else None
        last_agent_output = agent_outputs.get(last_agent_name)

        summary_text = None
        if last_agent_output is not None:
            user_prompt = json.dumps(last_agent_output, indent=2)
            response = await self._agent.run(user_prompt)
            summary_text = response.text or ""
            
        payload["last_agent_summary"] = summary_text

        return payload