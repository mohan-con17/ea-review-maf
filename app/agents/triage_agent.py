from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from app.domain.review_models import (
    ReviewSessionContext,
    DemographicsResult,
    ImageAnalysisResult,
)

logger = logging.getLogger(__name__)


class TriageExtraction(BaseModel):
    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
        "protected_namespaces": (),
    }


class TriageAgent:
    """
    TriageAgent performs a lossless merge of:
    - DemographicsResult
    - ImageAnalysisResult.image_components_json

    This agent MUST NEVER throw.
    """

    async def run(self, ctx: ReviewSessionContext) -> TriageExtraction:
        review_id = ctx.review_id

        try:
            demographics: Optional[DemographicsResult] = getattr(
                ctx, "demographics_from_json", None
            )
            image_analysis: Optional[ImageAnalysisResult] = getattr(
                ctx, "image_analysis", None
            )

            if not demographics and not image_analysis:
                logger.warning(
                    "[%s] TriageAgent: No demographics or image analysis available",
                    review_id,
                )
                result = TriageExtraction(
                    notes="No demographics or image analysis data found"
                )
                ctx.triage_results = result
                return result

            triage_dict: Dict[str, Any] = {}

            # Merge demographics
            if demographics:
                if hasattr(demographics, "__dict__"):
                    triage_dict.update(vars(demographics))
                else:
                    logger.warning("[%s] Invalid demographics object", review_id)

            # Merge image components
            if image_analysis and image_analysis.image_components_json:
                if isinstance(image_analysis.image_components_json, dict):
                    triage_dict.update(image_analysis.image_components_json)
                else:
                    logger.warning("[%s] image_components_json not a dict", review_id)

            try:
                result = TriageExtraction(**triage_dict)
            except ValidationError as ve:
                logger.error("[%s] TriageExtraction validation failed: %s", review_id, ve)
                result = TriageExtraction(notes="Triage validation failed")
                result.error = str(ve)

            ctx.triage_results = result
            return result

        except Exception as e:
            logger.exception("[%s] TriageAgent failed", review_id)
            result = TriageExtraction(notes="Triage processing failed")
            result.error = str(e)
            ctx.triage_results = result
            return result
