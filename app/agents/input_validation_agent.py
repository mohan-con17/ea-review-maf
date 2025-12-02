from __future__ import annotations

import base64
import logging
from typing import List, Any, Dict, Optional

from app.domain.review_models import (
    ReviewSessionContext,
    InputValidationResult,
    ValidationIssue,
)

logger = logging.getLogger(__name__)

class InputValidationAgent:

    async def validate(self, ctx: ReviewSessionContext) -> InputValidationResult:
        issues: List[ValidationIssue] = []

        metadata: Any = ctx.metadata
        
        logger.info(
            "Input Validation started", list(ctx.metadata.keys()) if isinstance(ctx.metadata, dict) else None
        )

        # ---- 1) metadata must be a JSON object ----
        if not isinstance(metadata, dict):
            issues.append(
                ValidationIssue(
                    field="metadata",
                    message="Metadata must be a JSON object.",
                    level="error",
                )
            )
            # If metadata itself is not a dict, we can't go further
            result = InputValidationResult(is_valid=False, issues=issues)
            ctx.validation_result = result
            return result

        # ---- 2) arch_img_url must exist and be a string ----
        raw_img_value: Optional[Any] = metadata.get("arch_img_url")
        
        logger.info("Image Validation started, image_base64 length=%s")

        if not isinstance(raw_img_value, str) or not raw_img_value.strip():
            issues.append(
                ValidationIssue(
                    field="metadata.arch_img_url",
                    message=(
                        "Architecture image is required and must be provided as a "
                        "base64-encoded string in 'arch_img_url'."
                    ),
                    level="error",
                )
            )
        else:
            # ---- 3) Strip data URL prefix if present ----
            # e.g. "data:image/png;base64,XXXXX"
            base64_str = raw_img_value.strip()
            if base64_str.startswith("data:"):
                comma_idx = base64_str.find(",")
                if comma_idx != -1:
                    base64_str = base64_str[comma_idx + 1 :]

            # ---- 4) Try to decode base64 ----
            try:
                img_bytes = base64.b64decode(base64_str, validate=True)
                if not img_bytes:
                    issues.append(
                        ValidationIssue(
                            field="metadata.arch_img_url",
                            message="Decoded architecture image is empty.",
                            level="error",
                        )
                    )
            except Exception:
                issues.append(
                    ValidationIssue(
                        field="metadata.arch_img_url",
                        message=(
                            "arch_img_url is not valid base64-encoded data. "
                            "If you're sending a data URL, it should look like "
                            "'data:image/png;base64,<BASE64>'."
                        ),
                        level="error",
                    )
                )

        # ---- final result ----
        is_valid = not any(str(issue.level).strip().lower() == "error" for issue in issues)
        result = InputValidationResult(is_valid=is_valid, issues=issues)
        ctx.validation_result = result
        return result
