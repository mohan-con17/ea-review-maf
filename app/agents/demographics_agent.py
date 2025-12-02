from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, DemographicsResult

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
    def __init__(self) -> None:
        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )

        instructions = (
             """
            You are the Demographics Agent in an enterprise architecture review workflow
            for a regulated BFSI environment.

            This section is typically a list of rows where each row has:
              - a label describing a data point
              - an answer value selected or filled by the project team.

            Your job is to interpret this "Project Specifics" content and infer
            the following normalized fields:

            - users:
                High-level primary user category for the application.
                Examples: "Internal", "External", "Mixed", "Partners", "Vendors".
            - sub_users:
                More specific user subtype if applicable.
                Examples: "Branch", "User", "Customer", "Operations", "Partner".
            - network:
                Primary access / connectivity mode for the application.
                Examples: "Internet", "Internet - VPN", "Internet - VDI",
                          "MPLS", "Intranet only", "Branch Network".
            - deployment:
                Deployment model at a high level.
                Allowed values (if possible): "cloud", "on prem", "hybrid".
                If unclear, choose the closest reasonable value or leave null.
            - cloud_provider:
                If deployment is cloud, infer the likely provider.
                Examples: "Azure", "AWS", "GCP", "OCI", "NA".
                Use "NA" or null if not cloud or unclear.
            - tier:
                BCM / criticality tier for the application.
                Allowed values if known: "1", "2", "3", "4", "5".
                If the tier is expressed as "Tier 3" or "BCM Tier 2",
                normalize to just "3" or "2". If unclear, leave null.

            You must respond with a SINGLE JSON object with this exact schema:

            {
              "users": string or null,
              "sub_users": string or null,
              "network": string or null,
              "deployment": string or null,
              "cloud_provider": string or null,
              "tier": string or null,
              "notes": string or null
            }

            Rules:
            - If you cannot infer a field, use null.
            - Do NOT invent values that have no basis in the input.
            - Do NOT include any additional top-level fields.
            - Do NOT include markdown, comments, or narrative outside the JSON.
            
            Output should be a single JSON only. No backticks or markdown.
            """
        )

        self._agent = ChatAgent(
            chat_client=chat_client,
            instructions=instructions,
            name="DemographicsAgent",
            max_output_tokens=1024,
            temperature=0.4,
        )

    async def run(self, ctx: ReviewSessionContext) -> DemographicsResult:

        # Directly access Project Specifics (always present)
        project_specifics = ctx.metadata.get("Project Specifics") or ctx.metadata.get("Project_Specifics")

        logger.info(
            f"[{ctx.review_id}] DemographicsAgent started | "
        )

        ps_json = json.dumps(project_specifics, ensure_ascii=False, indent=2)

        user_prompt = (
            "You are given the 'Project Specifics' section of a project metadata JSON.\n\n"
            "PROJECT_SPECIFICS:\n"
            f"{ps_json}\n\n"
        )

        # Call LLM
        response = await self._agent.run(user_prompt)
        raw_text = response.text or ""

        logger.info(
            f"[{ctx.review_id}] DemographicsAgent LLM call completed | "
            f"response_length={raw_text}"
        )

        try:
            parsed_dict = json.loads(raw_text)
            extraction = DemographicsExtraction.parse_obj(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(
                f"[{ctx.review_id}] DemographicsAgent failed to parse LLM response: {e}"
            )
            result = DemographicsResult(
                users=None,
                sub_users=None,
                network=None,
                deployment=None,
                cloud_provider=None,
                tier=None,
                error=str(e)
            )
            ctx.demographics_from_json = result
            return result

        # Build final result WITHOUT raw_extracted
        result = DemographicsResult(
            users=extraction.users,
            sub_users=extraction.sub_users,
            network=extraction.network,
            deployment=extraction.deployment,
            cloud_provider=extraction.cloud_provider,
            tier=extraction.tier,
        )
        
        ctx.demographics_from_json = result
        
        print("                                                                                      ")
        print("-----------------------------------Demographics Outupt-----------------------------------")
        print(result)
        print("                                                                                      ")
        
        return result