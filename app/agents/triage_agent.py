from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, DemographicsResult, ImageAnalysisResult

logger = logging.getLogger(__name__)

class TriageExtraction(BaseModel):
    fields: Dict[str, Any] = {}

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        # THIS makes the difference 👇
        "protected_namespaces": (),
    }

class TriageAgent:
    """
    TriageAgent merges Demographics and Image Analysis outputs into a unified JSON
    using an LLM (AzureOpenAIChatClient).
    """

    def __init__(self) -> None:
        chat_client = AzureOpenAIChatClient(
            deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )

        instructions = (
            """
            You are the Triage Agent in an enterprise architecture review workflow.

            You are given two JSON objects:
            1) demographics JSON with fields like users, sub_users, network, deployment, cloud_provider, tier
            2) image analysis JSON including labels extracted from architecture diagrams

            Your job is to:
            - Correlate the two JSONs accurately.
            - Derive the following fields explicitly or implicitly:
              "user", "sub_users", "network type", "deployment", "cloud provider",
              "if cloud, communicating directly via internet or via on prem", "tier"
            - Append all keys and values from the labels nested JSON in the image analysis JSON
            - Only include keys present in the input JSONs; do not assume extra fields.
            - For image_components_json consume the json directly rather than nesting it

            Respond with a single JSON object only. Example structure:

            {
              "user": string or null,
              "sub_users": string or null,
              "network type": string or null,
              "deployment": string or null,
              "cloud provider": string or null,
              "if cloud, communicating directly via internet or via on prem": string or null,
              "tier": string or null,
              "image_components_json": values
            }

            - No markdown.
            - No extra fields.
            - Use null if you cannot infer a field.
            """
        )

        self._agent = ChatAgent(
            chat_client=chat_client,
            instructions=instructions,
            name="TriageAgent",
            max_output_tokens=2048,
            temperature=0.2,
        )

    async def run(self, ctx: ReviewSessionContext) -> TriageExtraction:
        demographics: Optional[DemographicsResult] = getattr(ctx, "demographics_from_json", None)
        image_analysis: Optional[ImageAnalysisResult] = getattr(ctx, "image_analysis", None)

        if not demographics and not image_analysis:
            logger.warning(f"[{ctx.review_id}] TriageAgent: No input data available")
            triage_result = TriageExtraction(fields={"notes": "No demographics or image analysis data found"})
            ctx.triage_results = triage_result
            return triage_result

        # Prepare input JSONs
        demographics_dict = {
            "user": demographics.users if demographics else None,
            "sub_users": demographics.sub_users if demographics else None,
            "network type": demographics.network if demographics else None,
            "deployment": demographics.deployment if demographics else None,
            "cloud provider": demographics.cloud_provider if demographics else None,
            "tier": demographics.tier if demographics else None
        }
        demographics_json = json.dumps(demographics_dict if demographics else {}, indent=2)
        image_analysis_json = json.dumps(image_analysis.image_components_json if image_analysis else {}, indent=2)

        # Construct LLM prompt
        prompt = (
            f"Demographics JSON:\n{demographics_json}\n\n"
            f"Image Analysis JSON:\n{image_analysis_json}\n\n"
        )

        logger.info(f"[{ctx.review_id}] TriageAgent: Sending prompt to LLM")

        # Call LLM
        response = await self._agent.run(prompt)
        raw_text = response.text or ""

        logger.info(f"[{ctx.review_id}] TriageAgent LLM call completed | response_length={len(raw_text)}")

        # Parse JSON safely
        try:
            parsed_dict = json.loads(raw_text)
            triage_result = TriageExtraction.parse_obj(parsed_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[{ctx.review_id}] TriageAgent failed to parse LLM response: {e}")
            triage_result = TriageExtraction(fields={"notes": f"Error parsing LLM output: {str(e)}", "raw_output": raw_text})
            
        ctx.triage_results = triage_result
        
        print("                                                                                      ")
        print("-----------------------------------Triage Outupt-----------------------------------")
        print(triage_result)
        print("                                                                                      ")
        
        return triage_result