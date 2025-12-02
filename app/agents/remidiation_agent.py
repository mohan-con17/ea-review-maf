from __future__ import annotations

import json
from json import JSONDecodeError
import logging
from typing import Dict, Any, List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from pydantic import BaseModel

from app.config.settings import settings
from app.domain.review_models import ReviewSessionContext, RemediationSnapshot

logger = logging.getLogger(__name__)

def triage_to_json(triage: BaseModel) -> dict:

    # Standard fields
    base = triage.dict(exclude_none=True) if hasattr(triage, "dict") else triage.model_dump()

    result = {}

    # include base (fields attribute optional)
    for k, v in base.items():
        if k != "fields" and v not in (None, "", {}, []):
            result[k.replace(" ", "_")] = v

    # include dynamically added fields
    for attr in triage.__dict__.keys():
        if attr not in result and attr not in ("fields", "__pydantic_fields__", "__fields_set__"):
            value = getattr(triage, attr)
            if value not in (None, "", {}):
                result[attr.replace(" ", "_")] = value

    return result


# =====================================================================================
#                                 REMEDIATION AGENT
# =====================================================================================
class RemediationAgent:

    def __init__(self):
        # Azure Search client
        self._search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=settings.AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
        )

        # --------------------------------------------------------
        # LLM Agent for template selection
        # --------------------------------------------------------
        selection_instructions = """
        You are a Remediation Template Selection Agent.

        INPUT:
        - "context": triage JSON containing fields:
            user, sub_users, network type, deployment, cloud provider, tier
        - "candidate_templates": list of JSON from Azure Search

        TASK:
        - Perform a semantic match between context and candidate_templates.
        - Choose the BEST template based on meaning, not exact string match.
        - If no good match, pick index 0.
        - Respond ONLY with JSON:
          {
            "best_index": number
          }

        No explanations. No text outside JSON.
        """

        self._selection_agent = ChatAgent(
            chat_client=AzureOpenAIChatClient(deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME),
            instructions=selection_instructions,
            name="RemediationSelectionAgent",
            max_output_tokens=2048,
            temperature=0.0,
        )

        # --------------------------------------------------------
        # LLM Agent for semantic comparison
        # --------------------------------------------------------
        comparison_instructions = """
            You are a Semantic JSON Comparison Agent with deep expertise in Banking and Financial Services architectures, 
            for both Cloud (AWS, Azure, GCP) and On-Prem.

            INPUT:
            - "triage": JSON extracted from an existing architecture (observed view).
            - "template": JSON template used as the Standard / Target architecture (source of truth).

            The template is ALWAYS the canonical reference.
            The triage JSON is expected to be a partial subset of the template.

            IMPORTANT RULES:
            - Always compare the common top-level keys (e.g., Users, Sub users, Network, Deployment, Cloud provider, Tier, Akamai, WAF, DC, DR).
            - When the application is deployed on cloud("Azure", "AWS", "OCI") then the below rules to be followed
                Determine the traffic type from "triage":
                - If the triage explicitly states or implies that traffic is coming **directly** (not via on-prem), then:
                    - Compare the triage JSON with the template JSON's top-level keys AND the nested object under `"communicate directly"`.
                    - Ignore the nested object under `"via on prem"` completely.
                - If the triage explicitly states or implies that traffic is coming **via on-prem**, then:
                    - Compare the triage JSON with the template JSON's top-level keys AND the nested object under `"via on prem"`.
                    - Ignore the nested object under `"communicate directly"` completely.
                - If in the template JSON does not contain any nested json, then:
                    - Compare the triage JSON with the template JSON's all key value pairs.
                - For inference:
                    - Phrases like "direct connection", "no on-prem", "direct to cloud", or absence of VPN/Direct Connect indicate **communicate directly**.
                    - Phrases like "through on-prem", "via data center", "VPN", "Direct Connect" indicate **via on-prem traffic**.
            - Do not count "combination_path in available_components, missing_components and similarity_percent calculations.
            - Treat keys case-insensitively when comparing (e.g., "Users" ≈ "users").
            - You are an experienced banking and financial domain architect: use semantic understanding of components 
            (e.g., gateways, firewalls, DMZ, CDN, WAF, monitoring, DR, etc.), not just keyword string matching.

            TASKS:

            1. Semantically compare triage and template:
                - Match key+value pairs by meaning, not exact string.
                - Example semantic matches:
                    - "AWS EC2" ≈ "Compute on AWS" ≈ "Compute_Instances"
                    - "user" ≈ "Users" ≈ "end_users"
                    - "WAF" ≈ "Web Application Firewall"
                    - "DR Region" ≈ "Disaster Recovery"
                    - "DC Region" ≈ "Data Center"

            2. Determine:
                - available_components_in_triage:
                    List of string identifiers for key+value pairs in TEMPLATE that have a
                    semantically matching key+value in TRIAGE.
                    Use a consistent string format such as "path.to.key = value".
                    EXCLUDE any key whose name is "combination_path" (case insensitive).

                - missing_components_in_triage:
                    List of string identifiers for key+value pairs that exist in TEMPLATE but
                    do NOT have any semantically matching key+value in TRIAGE.
                    Also EXCLUDE any key whose name is "combination_path" (case insensitive).
                    
                - missing_components_in_template:
                    List of string identifiers for key+value pairs that exist in TRIAGE but
                    do NOT have any semantically matching key+value in TEMPLATE.

            3. Compute similarity_percent:
                - Let T = total number of TEMPLATE key+value pairs considered,
                    EXCLUDING any key named "combination_path" (case insensitive).
                - Let M = number of those TEMPLATE key+value pairs that are in available_components.
                - If T == 0, similarity_percent = 0.
                - Else, similarity_percent = round(100 * M / T).

            OUTPUT FORMAT:
            Return ONLY a single valid JSON object. 
            Do NOT include any backticks, markdown, comments, or extra text. 
            Do NOT wrap the JSON in ``` fences. Output must start with '{' and end with '}'.

            The JSON MUST have exactly these keys:

            {
                "available_components_in_triage": [],
                "missing_components_in_triage": [],
                "similarity_percent": 0,
                "missing_components_in_template": []
            }
        """ 

        self._comparison_agent = ChatAgent(
            chat_client=AzureOpenAIChatClient(deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME),
            instructions=comparison_instructions,
            name="RemediationComparisonAgent",
            max_output_tokens=2048,
            temperature=0.4,
        )

    # ==============================================================================
    #                                 PUBLIC ENTRYPOINT
    # ==============================================================================
    async def run(self, ctx: ReviewSessionContext) -> RemediationSnapshot:
        review_id = ctx.review_id

        # -----------------------------
        # Debug print all relevant inputs
        # -----------------------------
        triage = getattr(ctx, "triage_results", None)

        if not triage:
            logger.warning(f"[{review_id}] No triage results; skipping remediation")
            snapshot = RemediationSnapshot()
            ctx.remediation_result = snapshot
            return snapshot

        # ----------------------------------------------------------------------
        # STEP 1: Azure Search using triage values directly
        # ----------------------------------------------------------------------
        # Extract only the relevant fields from triage for search
        triage_json = {
            "user": getattr(triage, "user", None),
            "sub_users": getattr(triage, "sub_users", None),
            "network type": getattr(triage, "network type", None),
            "deployment": getattr(triage, "deployment", None),
            "cloud provider": getattr(triage, "cloud provider", None),
            "tier": getattr(triage, "tier", None),
        }

        search_text = " ".join([str(v) for v in triage_json.values() if v])
        candidate_templates = self._search_templates(search_text)
        logger.info(f"[{review_id}] Retrieved {len(candidate_templates)} candidate templates")

        # LLM selects best template
        best_index = await self._select_best_template(triage_json, candidate_templates)
        template_json = candidate_templates[best_index] if best_index is not None and candidate_templates else {}

        # ----------------------------------------------------------------------
        # STEP 2: LLM Semantic Comparison
        # ----------------------------------------------------------------------
        triage_cleaned_json = triage_to_json(triage)
        comparison = await self._compare_semantic(triage_cleaned_json, template_json)

        # ----------------------------------------------------------------------
        # Build Snapshot
        # ----------------------------------------------------------------------

        snapshot = RemediationSnapshot(
            combination_path=json.loads(template_json.get("chunk")).get("combination_path"),
            template=template_json,
            missing_components_in_triage=comparison.get("missing_components_in_triage", []),
            available_components_in_triage=comparison.get("available_components_in_triage", []),
            missing_components_in_template=comparison.get("missing_components_in_template", []),
            similarity_percent=comparison.get("similarity_percent"),
        )

        ctx.remediation_result = snapshot
        
        metadata = getattr(ctx, "metadata", {}) or {}
        metadata["similarity_score"] = comparison.get("similarity_percent", 0)

        ctx.computed_metadata = metadata
        
        print("                                                                                      ")
        print("-----------------------------------Remediation Output-----------------------------------")
        print("JSON Template", snapshot.combination_path)
        print(snapshot.template)
        print("                                                                                      ")
        print("Missing in Triage", snapshot.missing_components_in_triage)
        print("                                                                                      ")
        print("Available in Triage", snapshot.available_components_in_triage)
        print("                                                                                      ")
        print("Missing in JSON Template", snapshot.missing_components_in_template)
        print("                                                                                      ")
        print("Similarity Score", snapshot.similarity_percent)
        print("                                                                                      ")

        return snapshot


    # ==============================================================================
    #                                 INTERNAL METHODS
    # ==============================================================================

    def _search_templates(self, search_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self._search_client.search(search_text=search_text, top=top_k)
            return [dict(doc) for doc in results]
        except Exception as e:
            logger.error(f"Azure Search failed: {e}")
            return []

    async def _select_best_template(self, triage_json: Dict[str, Any], candidates: List[Dict[str, Any]]) -> int:
        payload = {"context": triage_json, "candidate_templates": candidates}
        response = await self._selection_agent.run(json.dumps(payload, indent=2))

        try:
            data = json.loads(response.text or "{}")
            return data.get("best_index", 0)
        except Exception:
            return 0

    async def _compare_semantic(self, triage_json: Dict[str, Any], template_json: Dict[str, Any]) -> Dict[str, Any]:
        
        raw_chunk = template_json["chunk"]         # string that looks like JSON
        try:
            cleaned_json_template = json.loads(raw_chunk)  # now it's a dict
        except JSONDecodeError:
            cleaned_json_template = {}                     # or handle error as you like

        payload = {"triage": triage_json, "template": cleaned_json_template}
        response = await self._comparison_agent.run(json.dumps(payload, indent=2))

        try:
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                # remove possible language tag like ```json
                first_brace = raw.find("{")
                raw = raw[first_brace:]
            data = json.loads(raw)

            return data
        except Exception:
            return {
                "missing_components_in_triage": [],
                "available_components_in_triage": [],
                "missing_components_in_template": [],
                "similarity_percent": 0
            }
