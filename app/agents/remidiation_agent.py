from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from typing import Dict, Any, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from pydantic import BaseModel

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, RemediationSnapshot
from app.prompts.prompt_registry import PromptRegistry
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# TRIAGE NORMALIZATION (RESTORED)
# ------------------------------------------------------------------
def triage_to_json(triage: BaseModel) -> Dict[str, Any]:
    base = (
        triage.dict(exclude_none=True)
        if hasattr(triage, "dict")
        else triage.model_dump(exclude_none=True)
    )

    result: Dict[str, Any] = {}

    for k, v in base.items():
        if k != "fields" and v not in (None, "", {}, []):
            result[k.replace(" ", "_")] = v

    for attr, value in triage.__dict__.items():
        if attr.startswith("_"):
            continue
        if attr not in result and value not in (None, "", {}, []):
            result[attr.replace(" ", "_")] = value

    return result


class RemediationAgent:
    """
    RemediationAgent
    ----------------
    - Azure Search + LLM based remediation
    - HARD FAILS on any meaningful error
    """

    def __init__(self):
        self._search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
        )

        self._token_counter = TokenCounter()

        selection_prompt = PromptRegistry.get(
            "remediation_template_selection", "v1"
        )
        comparison_prompt = PromptRegistry.get(
            "remediation_semantic_comparison", "v1"
        )

        self._selection_agent = ChatAgent(
            chat_client=AzureOpenAIChatClient(
                deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            ),
            instructions=selection_prompt["messages"]["system"],
            name="RemediationSelectionAgent",
            max_output_tokens=selection_prompt["model"]["max_tokens"],
            temperature=selection_prompt["model"]["temperature"],
        )

        self._comparison_agent = ChatAgent(
            chat_client=AzureOpenAIChatClient(
                deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            ),
            instructions=comparison_prompt["messages"]["system"],
            name="RemediationComparisonAgent",
            max_output_tokens=comparison_prompt["model"]["max_tokens"],
            temperature=comparison_prompt["model"]["temperature"],
        )

    # ==============================================================
    # PUBLIC ENTRYPOINT
    # ==============================================================

    async def run(self, ctx: ReviewSessionContext) -> RemediationSnapshot:
        review_id = ctx.review_id

        try:
            triage_obj = getattr(ctx, "triage_results", None)
            if not triage_obj:
                raise RuntimeError("Triage results missing for remediation")

            triage_json = triage_to_json(triage_obj)
            if not triage_json:
                raise RuntimeError("Triage results empty after normalization")

            # ----------------------------------------------------------
            # STEP 1: Azure Search
            # ----------------------------------------------------------
            search_text = " ".join(str(v) for v in triage_json.values())
            templates = self._search_templates(search_text)

            if not templates:
                raise RuntimeError("No remediation templates found")

            # ----------------------------------------------------------
            # STEP 2: Template selection
            # ----------------------------------------------------------
            best_index = await self._select_best_template(
                ctx, triage_json, templates
            )

            if best_index is None or best_index >= len(templates):
                raise RuntimeError("Invalid remediation template index")

            template = templates[best_index]

            # ----------------------------------------------------------
            # STEP 3: Semantic comparison
            # ----------------------------------------------------------
            comparison = await self._compare_semantic(
                ctx, triage_json, template
            )

            if "similarity_percent" not in comparison:
                raise RuntimeError("Comparison output missing similarity_percent")

            # ----------------------------------------------------------
            # STEP 4: Snapshot
            # ----------------------------------------------------------
            try:
                combination_path = json.loads(
                    template.get("chunk", "{}")
                ).get("combination_path")
            except Exception:
                combination_path = None

            snapshot = RemediationSnapshot(
                combination_path=combination_path,
                template=template,
                missing_components_in_triage=comparison.get(
                    "missing_components_in_triage", []
                ),
                available_components_in_triage=comparison.get(
                    "available_components_in_triage", []
                ),
                missing_components_in_template=comparison.get(
                    "missing_components_in_template", []
                ),
                similarity_percent=comparison["similarity_percent"],
            )

            ctx.remediation_result = snapshot
            ctx.computed_metadata["similarity_score"] = snapshot.similarity_percent

            return snapshot

        except Exception as e:
            logger.exception("[%s] RemediationAgent failed", review_id)

            snapshot = RemediationSnapshot()
            snapshot.error = str(e)
            ctx.remediation_result = snapshot
            return snapshot

    # ==============================================================
    # INTERNAL METHODS
    # ==============================================================

    def _search_templates(self, search_text: str) -> List[Dict[str, Any]]:
        try:
            results = self._search_client.search(
                search_text=search_text, top=5
            )
            return [dict(doc) for doc in results]
        except Exception as e:
            raise RuntimeError(f"Azure Search failed: {e}")

    async def _select_best_template(
        self,
        ctx: ReviewSessionContext,
        triage_json: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> int:

        prompt = json.dumps(
            {"context": triage_json, "candidate_templates": candidates},
            indent=2,
        )

        ctx.last_input_tokens = self._token_counter.count_text(prompt)

        response = await self._selection_agent.run(prompt)
        output_text = response.text or ""

        ctx.last_output_tokens = self._token_counter.count_text(output_text)
        ctx.last_total_tokens = (
            ctx.last_input_tokens + ctx.last_output_tokens
        )

        ctx.llm_traces.append({
            "agent": "remediation",
            "prompt": prompt,
            "response": output_text,
            "status": "success",
        })

        raw = output_text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):]

        try:
            parsed = json.loads(raw)
            return int(parsed["best_index"])
        except Exception:
            ctx.llm_traces.append({
                "agent": "remediation",
                "prompt": prompt,
                "response": output_text,
                "status": "failure",
            })
            raise RuntimeError("Template selection parsing failed")

    async def _compare_semantic(
        self,
        ctx: ReviewSessionContext,
        triage_json: Dict[str, Any],
        template: Dict[str, Any],
    ) -> Dict[str, Any]:

        try:
            template_clean = json.loads(template.get("chunk", "{}"))
        except Exception:
            raise RuntimeError("Template chunk is not valid JSON")

        prompt = json.dumps(
            {"triage": triage_json, "template": template_clean},
            indent=2,
        )

        ctx.last_input_tokens = self._token_counter.count_text(prompt)

        response = await self._comparison_agent.run(prompt)
        output_text = response.text or ""

        ctx.last_output_tokens = self._token_counter.count_text(output_text)
        ctx.last_total_tokens = (
            ctx.last_input_tokens + ctx.last_output_tokens
        )

        ctx.llm_traces.append({
            "agent": "remediation",
            "prompt": prompt,
            "response": output_text,
            "status": "success",
        })

        raw = output_text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):]

        try:
            parsed = json.loads(raw)
            similarity = parsed.get("similarity_percent", 0)
            if isinstance(similarity, str):
                similarity = int(similarity.replace("%", "").strip())
            parsed["similarity_percent"] = similarity
            return parsed
        except Exception:
            ctx.llm_traces.append({
                "agent": "remediation",
                "prompt": prompt,
                "response": output_text,
                "status": "failure",
            })
            raise RuntimeError("Semantic comparison parsing failed")
