from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, ValidationError
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, DemographicsResult
from app.prompts.prompt_registry import PromptRegistry
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DemographicsExtraction(BaseModel):
    users: Optional[str] = None
    sub_users: Optional[str] = None
    network: Optional[str] = None
    deployment: Optional[str] = None
    cloud_provider: Optional[str] = None
    tier: Optional[str] = None
    notes: Optional[str] = None


class DemographicsAgent:
    """
    LLM-based demographics extraction agent.
    HARD FAILS on LLM or parsing issues.
    """

    def __init__(self) -> None:
        self._token_counter = TokenCounter()

        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )

        prompt = PromptRegistry.get("demographics_extraction", "v1")

        self._agent = ChatAgent(
            chat_client=chat_client,
            instructions=prompt["messages"]["system"],
            name="DemographicsAgent",
            max_output_tokens=prompt["model"]["max_tokens"],
            temperature=prompt["model"]["temperature"],
        )

        self._user_template = prompt["messages"]["user_template"]

    async def run(self, ctx: ReviewSessionContext) -> DemographicsResult:
        review_id = ctx.review_id
        user_prompt = ""
        output_text = ""

        try:
            project_specifics = (
                ctx.metadata.get("Project Specifics")
                or ctx.metadata.get("Project_Specifics")
            )

            if not project_specifics:
                raise RuntimeError("Project specifics missing for demographics extraction")

            ps_json = json.dumps(project_specifics, ensure_ascii=False, indent=2)
            user_prompt = self._user_template.replace("{{project_specifics}}", ps_json)

            # ---------------- Token counting (input) ----------------
            ctx.last_input_tokens = self._token_counter.count_text(user_prompt)

            response = await self._agent.run(user_prompt)
            output_text = response.text or ""

            # ---------------- Token counting (output) ----------------
            ctx.last_output_tokens = self._token_counter.count_text(output_text)
            ctx.last_total_tokens = (
                ctx.last_input_tokens + ctx.last_output_tokens
            )

            # ---------------- LLM trace (SUCCESS) ----------------
            ctx.llm_traces.append({
                "agent": "demographics",
                "prompt": user_prompt,
                "response": output_text,
                "status": "success",
            })

            parsed = json.loads(output_text)
            extraction = DemographicsExtraction.parse_obj(parsed)

            result = DemographicsResult(
                users=extraction.users,
                sub_users=extraction.sub_users,
                network=extraction.network,
                deployment=extraction.deployment,
                cloud_provider=extraction.cloud_provider,
                tier=extraction.tier,
            )

            ctx.demographics_from_json = result
            return result

        except (json.JSONDecodeError, ValidationError) as e:
            logger.exception("[%s] Demographics parsing failed", review_id)

            # ---------------- LLM trace (FAILURE) ----------------
            ctx.llm_traces.append({
                "agent": "demographics",
                "prompt": user_prompt,
                "response": output_text,
                "status": "failure",
            })

            result = DemographicsResult()
            result.error = "Failed to parse demographics response"
            ctx.demographics_from_json = result
            return result

        except Exception as e:
            logger.exception("[%s] DemographicsAgent failed", review_id)

            # ---------------- LLM trace (FAILURE) ----------------
            ctx.llm_traces.append({
                "agent": "demographics",
                "prompt": user_prompt,
                "response": output_text,
                "status": "failure",
            })

            result = DemographicsResult()
            result.error = str(e)
            ctx.demographics_from_json = result
            return result
