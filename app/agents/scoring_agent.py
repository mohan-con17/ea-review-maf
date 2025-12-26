# from __future__ import annotations

# import json
# import logging
# from typing import Dict, Any

# from pydantic import BaseModel, ValidationError
# from agent_framework import ChatAgent
# from agent_framework.azure import AzureOpenAIChatClient

# from app.config.settings import settings
# from app.domain.review_models import AgentScore

# logger = logging.getLogger(__name__)


# # ==============================================================
# # STRICT SCHEMA (1–10 SCALE)
# # ==============================================================

# class JudgeSchema(BaseModel):
#     accuracy: int
#     bias: int
#     hallucination: int
#     confidence: int
#     notes: Dict[str, str]


# # ==============================================================
# # SCORING AGENT
# # ==============================================================

# class ScoringAgent:
#     """
#     Observational LLM Judge Agent.

#     - Scores on a scale of 1–10
#     - NEVER affects review outcome
#     - Caller controls sequencing
#     """

#     SYSTEM_PROMPT = """
#         You are an expert AI response evaluator.

#         Evaluate how well the RESPONSE follows the PROMPT.

#         Scoring rules:
#         - Accuracy: prompt adherence & missing information. Higher score is considered good.
#         - Bias: cloud, technology, architecture, organization, or generic. Lower score is considered good.
#         - Hallucination/Relevance: invented or irrelevant content. Lower score is considered good.
#         - Confidence: overconfidence or excessive hedging. Higher score is considered good.

#         Rules:
#         - Scores must be integers between 1 and 10
#         - 1 = very poor, 10 = excellent
#         - Missing information is an accuracy issue, NOT hallucination
#         - Return ONLY valid JSON
#     """

#     USER_TEMPLATE = """
#         Agent Name: {agent}
#         Execution Status: {status}

#         PROMPT:
#         {prompt}

#         RESPONSE:
#         {response}

#         Return JSON exactly in this format:
#         {{
#         "accuracy": 1,
#         "bias": 1,
#         "hallucination": 1,
#         "confidence": 1,
#         "notes": {{
#             "accuracy": "",
#             "bias": "",
#             "hallucination": "",
#             "confidence": ""
#         }}
#         }}
#     """

#     def __init__(self) -> None:
#         self._agent = ChatAgent(
#             chat_client=AzureOpenAIChatClient(
#                 deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
#             ),
#             instructions=self.SYSTEM_PROMPT,
#             name="LLMScoringAgent",
#             temperature=0.0,
#             max_output_tokens=1024,
#         )

#     # ----------------------------------------------------------
#     # PUBLIC API
#     # ----------------------------------------------------------

#     async def score(self, trace: Dict[str, Any]) -> AgentScore:
#         user_prompt = self.USER_TEMPLATE.format(
#             agent=trace.get("agent"),
#             status=trace.get("status"),
#             prompt=trace.get("prompt"),
#             response=trace.get("response"),
#         )

#         try:
#             response = await self._agent.run(user_prompt)
#             raw_text = response.text or ""

#             parsed = json.loads(raw_text)
#             judge = JudgeSchema.parse_obj(parsed)

#             def clamp(val: int) -> int:
#                 return max(1, min(10, int(val)))

#             return AgentScore(
#                 accuracy=clamp(judge.accuracy),
#                 bias=clamp(judge.bias),
#                 hallucination=clamp(judge.hallucination),
#                 confidence=clamp(judge.confidence),
#                 notes=judge.notes,
#             )

#         except (json.JSONDecodeError, ValidationError) as e:
#             print("ScoringAgent schema error: %s", e)
#             raise

#         except Exception as e:
#             print("ScoringAgent execution failed", e)
#             raise


from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any

from pydantic import BaseModel, ValidationError
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import AgentScore

logger = logging.getLogger(__name__)


# ==============================================================
# STRICT SCHEMA (1–10 SCALE)
# ==============================================================

class JudgeSchema(BaseModel):
    accuracy: int
    bias: int
    hallucination: int
    confidence: int
    notes: Dict[str, str]


# ==============================================================
# SCORING AGENT
# ==============================================================

class ScoringAgent:
    """
    Observational LLM Judge Agent.

    - Scores on a scale of 1–10
    - NEVER affects review outcome
    - Caller controls sequencing
    """

    SYSTEM_PROMPT = """
        You are an expert AI response evaluator.

        Evaluate how well the RESPONSE follows the PROMPT.

        Scoring rules:
        - Accuracy: prompt adherence & missing information. Higher score is considered good.
        - Bias: cloud, technology, architecture, organization, or generic. Lower score is considered good.
        - Hallucination/Relevance: invented or irrelevant content. Lower score is considered good.
        - Confidence: overconfidence or excessive hedging. Higher score is considered good.

        Rules:
        - Scores must be integers between 1 and 10
        - 1 = very poor, 10 = excellent
        - Missing information is an accuracy issue, NOT hallucination
        - Return ONLY valid JSON
    """

    USER_TEMPLATE = """
        Agent Name: {agent}
        Execution Status: {status}

        PROMPT:
        {prompt}

        RESPONSE:
        {response}

        Return JSON exactly in this format:
        {{
        "accuracy": 1,
        "bias": 1,
        "hallucination": 1,
        "confidence": 1,
        "notes": {{
            "accuracy": "",
            "bias": "",
            "hallucination": "",
            "confidence": ""
        }}
        }}
    """

    def __init__(self) -> None:
        self._agent = ChatAgent(
            chat_client=AzureOpenAIChatClient(
                deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            ),
            instructions=self.SYSTEM_PROMPT,
            name="LLMScoringAgent",
            temperature=0.0,
            max_output_tokens=1024,
        )

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    async def score(self, trace: Dict[str, Any]) -> AgentScore:
        user_prompt = self.USER_TEMPLATE.format(
            agent=trace.get("agent"),
            status=trace.get("status"),
            prompt=trace.get("prompt"),
            response=trace.get("response"),
        )

        try:
            response = await self._agent.run(user_prompt)
            raw_text = response.text or ""

            # FIX: Remove Markdown code blocks (```json ... ```) which cause JSONDecodeError
            clean_json = re.sub(r"```json\s?|```", "", raw_text).strip()
            
            if not clean_json:
                raise ValueError("ScoringAgent received an empty response from the LLM")

            parsed = json.loads(clean_json)
            
            # Use model_validate for Pydantic V2 or keep the logic compatible with your env
            judge = JudgeSchema(**parsed)

            def clamp(val: int) -> int:
                """Ensures scores remain within the 1-10 range."""
                try:
                    return max(1, min(10, int(val)))
                except (TypeError, ValueError):
                    return 1

            return AgentScore(
                accuracy=clamp(judge.accuracy),
                bias=clamp(judge.bias),
                hallucination=clamp(judge.hallucination),
                confidence=clamp(judge.confidence),
                notes=judge.notes,
            )

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"ScoringAgent parsing error: {e}. Raw text was: {raw_text}")
            raise

        except Exception as e:
            logger.error(f"ScoringAgent execution failed: {e}")
            raise