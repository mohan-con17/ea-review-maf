from __future__ import annotations

import json
import logging
from typing import Dict, Any, List

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.orchestrator.agent_registry import get_agents_definition
from app.domain.review_models import ReviewSessionContext, PlanDecision

logger = logging.getLogger(__name__)

class PlannerAgent:

    def __init__(self) -> None:
        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )
        
        instructions = ( # Enhance the prompt with enums, different roles with proper descriptions
            """
            You are the **Planner** in an enterprise architecture review workflow for a regulated BFSI environment. 
            Your sole responsibility is to determine **which agents to run and in what sequence**, based strictly on 
            their declared dependencies.

            WHAT YOU ARE GIVEN
            You will receive a single JSON object as input, with the keys "available_sections" and "agents".
                - "available_sections" describes which metadata sections were provided in the submission. 
                    You do NOT need to validate or analyze these sections. They are only contextual for your reference 
                    to know what kind of data we are dealing with.
                - "agents" describes all available agents, their names, descriptions, and dependencies. 
                Use these descriptions to understand how each agent works and use the dependencies to determine 
                the correct execution order.

            WHAT YOU MUST DO
            Your task is to construct a **staged execution plan**, respecting:
            - All dependency constraints.
            - Possibility of parallel execution where dependencies allow it.

            Rules:
            - Agents with no dependencies may be placed together in the first stage.
            - An agent must appear only after all of its dependencies have appeared in earlier stages.
            - Do NOT infer new agents or rename any.
            - Only use the agents given in the input.

            OUTPUT FORMAT (STRICT & EXCLUSIVE)
            You MUST respond with a **single JSON object**, with NO surrounding text, like this:

            {
            "stages": [
                ["agent_1", "agent_3"],
                ["agent_2"],
                ["agent_4"]
            ],
            "notes": "short explanation of why this layout was chosen"
            }

            REQUIRED RULES:
            - The "stages" field must be an array of arrays of strings.
            - Each inner array is one stage; agents in that stage may run in parallel.
            - Stages are executed in order from first to last.
            - Every agent in the output must be one of the agents provided in the input JSON.
            - "notes" must be a short explanation (1–3 sentences).
            - No other top-level keys are allowed.
            - No prose, no markdown, and no comments outside the JSON object.

            Your entire response must be exactly and only that JSON structure.
            """
        )

        self._agent = ChatAgent(
            chat_client=chat_client,
            instructions=instructions,
            name="PlannerAgent",
            max_output_tokens=512,
            temperature=0.1,
        )

    async def plan(self, ctx: ReviewSessionContext) -> PlanDecision:
        metadata: Dict[str, Any] = ctx.metadata if isinstance(ctx.metadata, dict) else {}
        
        available_sections = list(metadata.keys())

        planner_input = {
            "available_sections": available_sections,
            "agents": get_agents_definition(),
        }

        user_prompt = (
            "Planner input:\n\n"
            f"{json.dumps(planner_input, ensure_ascii=False, indent=2)}\n\n"
            "Decide the staged layout and explanation. Respond ONLY with the JSON object in the format described in your instructions."
        )

        logger.debug(
            f"[{ctx.review_id}] Planner prompt length={len(user_prompt)}"
        )

        response = await self._agent.run(user_prompt)
        text = response.text or "{}"

        try:
            raw = json.loads(text)
        except Exception:
            logger.error(
                f"[{ctx.review_id}] Planner returned invalid JSON, using fallback layout",
                exc_info=True,
            )

            # Fallback layout reflecting new dependencies
            stages = [
                ["demographics", "image_analysis"],
                ["triage"],
                ["remediation"],
            ]
            notes = (
                "Fallback: demographics and image_analysis run in parallel; "
                "triage processes their outputs; remediation uses triage results."
            )
            return PlanDecision(stages=stages, notes=notes)

        stages_raw = raw.get("stages")
        notes = raw.get("notes")

        stages = self._normalize_stages(stages_raw)

        decision = PlanDecision(stages=stages, notes=notes)

        print("                                                                                      ")
        print("-----------------------------------Planner Decision-----------------------------------")
        print(decision)
        print("                                                                                      ")
        
        return decision

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _normalize_stages(self, stages_raw: Any) -> List[List[str]]:
        """
        Ensure stages is a list[list[str]] and only contains known agent names.
        """
        allowed = {"demographics", "image_analysis", "triage", "remediation"}

        if not isinstance(stages_raw, list):
            return []

        normalized: List[List[str]] = []
        for stage in stages_raw:
            if not isinstance(stage, list):
                continue
            agents = [a for a in stage if isinstance(a, str) and a in allowed]
            if agents:
                normalized.append(agents)

        return normalized
