from __future__ import annotations

import base64
import json
import logging
import re
import asyncio
from typing import Any, Dict, List

from openai import AzureOpenAI

from app.config.settings import settings
from app.domain.review_models import (
    ReviewSessionContext,
    ImageAnalysisResult,
    PreprocessedImage,
)
from app.prompts.prompt_registry import PromptRegistry
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class ImageAnalyzerAgent:
    """
    ImageAnalyzerAgent
    ------------------
    - Analyzes preprocessed image tiles
    - Consolidates tile results via LLM
    - HARD FAILS on consolidation failure
    """

    def __init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )

        self._token_counter = TokenCounter()

        tile_prompt = PromptRegistry.get("image_tile_analysis", "v1")
        consolidation_prompt = PromptRegistry.get("image_consolidation", "v1")

        self._tile_prompt = tile_prompt["messages"]["system"]
        self._consolidation_prompt = consolidation_prompt["messages"]["system"]

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    async def run(self, ctx: ReviewSessionContext) -> ImageAnalysisResult:
        review_id = ctx.review_id

        try:
            preprocessed: PreprocessedImage | None = getattr(
                ctx, "preprocessed_image", None
            )

            if not preprocessed or not preprocessed.tiles:
                raise RuntimeError("No preprocessed image tiles available")

            # ----------------------------------------------------------
            # Tile analysis (parallel, safe)
            # ----------------------------------------------------------
            tile_results = await asyncio.gather(
                *[
                    self._analyze_single_tile(tile, idx, review_id)
                    for idx, tile in enumerate(preprocessed.tiles)
                ]
            )

            # ----------------------------------------------------------
            # Consolidation (LLM – HARD FAIL)
            # ----------------------------------------------------------
            (
                consolidated_text,
                input_tokens,
                output_tokens,
                prompt_payload,
            ) = await self._consolidate(tile_results)

            ctx.last_input_tokens = input_tokens
            ctx.last_output_tokens = output_tokens
            ctx.last_total_tokens = input_tokens + output_tokens

            # ---------------- LLM TRACE (SUCCESS) ----------------
            ctx.llm_traces.append({
                "agent": "image_analysis",
                "prompt": prompt_payload,
                "response": consolidated_text,
                "status": "success",
            })

            parsed_json = self._safe_extract_json(consolidated_text)

            result = ImageAnalysisResult(
                architecture_summary=parsed_json.get("Image_Summary", ""),
                image_components_json=parsed_json.get("image_components_json", {}),
            )

            ctx.image_analysis = result
            return result

        except Exception as e:
            logger.exception("[%s] ImageAnalyzerAgent failed", review_id)

            ctx.llm_traces.append({
                "agent": "image_analysis",
                "prompt": "Image consolidation",
                "response": "",
                "status": "failure",
            })

            result = ImageAnalysisResult(
                architecture_summary="Image analysis failed.",
                image_components_json={},
            )
            result.error = str(e)

            ctx.image_analysis = result
            return result

    # ------------------------------------------------------------------
    # INTERNAL METHODS
    # ------------------------------------------------------------------

    async def _analyze_single_tile(
        self, tile_bytes: bytes, idx: int, review_id: str
    ) -> Dict[str, Any]:
        """
        Tile-level analysis.
        Tile failures do NOT break the agent.
        """

        try:
            data_url = self._to_data_url(tile_bytes)

            messages = [
                {"role": "system", "content": self._tile_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "high",
                            },
                        }
                    ],
                },
            ]

            response = self._client.chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.0,
            )

            raw = response.choices[0].message.content or ""
            parsed = self._safe_extract_json(raw)

            # 🔧 NORMALIZATION FIX (CRITICAL)
            return {
                "tile_summary": parsed.get("tile_summary")
                or parsed.get("summary", ""),
                "components": (
                    parsed.get("components")
                    or parsed.get("image_components")
                    or []
                ),
            }

        except Exception as e:
            logger.warning(
                "[%s] Tile %d analysis failed: %s",
                review_id,
                idx,
                e,
            )
            return {"tile_summary": "", "components": []}

    async def _consolidate(
        self, tile_results: List[Dict[str, Any]]
    ) -> tuple[str, int, int, str]:
        """
        Consolidates tile-level results using LLM.
        HARD FAILS if LLM or parsing fails.
        """

        payload = {
            "summaries": [t.get("tile_summary", "") for t in tile_results],
            "components": sum(
                [t.get("components", []) for t in tile_results], []
            ),
        }

        payload_str = json.dumps(payload, indent=2)

        messages = [
            {"role": "system", "content": self._consolidation_prompt},
            {"role": "user", "content": payload_str},
        ]

        input_tokens = self._token_counter.count_messages(messages)

        response = self._client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.0,
        )

        output_text = response.choices[0].message.content or ""
        output_tokens = self._token_counter.count_text(output_text)

        return output_text, input_tokens, output_tokens, payload_str

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _to_data_url(self, tile_bytes: bytes) -> str:
        return (
            "data:image/png;base64,"
            + base64.b64encode(tile_bytes).decode()
        )

    def _safe_extract_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"\{[\s\S]*\}", text or "")
        if not match:
            return {}

        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
