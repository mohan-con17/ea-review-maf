from __future__ import annotations

import base64
import logging
from dataclasses import asdict
from typing import List, Any, Optional

from app.domain.review_models import (
    ReviewSessionContext,
    InputValidationResult,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


class InputValidationAgent:
    """
    InputValidationAgent
    --------------------
    - Validates request metadata
    - NEVER throws
    - Always returns InputValidationResult
    """

    async def validate(self, ctx: ReviewSessionContext) -> InputValidationResult:
        review_id = ctx.review_id
        issues: List[ValidationIssue] = []

        try:
            metadata: Any = ctx.metadata

            logger.info(
                "[%s] Input validation started. Metadata keys=%s",
                review_id,
                list(metadata.keys()) if isinstance(metadata, dict) else None,
            )

            # 1️⃣ Metadata must be dict
            if not isinstance(metadata, dict):
                issues.append(
                    ValidationIssue(
                        field="metadata",
                        message="Metadata must be a JSON object.",
                        level="error",
                    )
                )
                result = InputValidationResult(is_valid=False, issues=issues)
                ctx.validation_result = result
                return result

            # 2️⃣ arch_img_url validation
            raw_img_value: Optional[Any] = metadata.get("arch_img_url")

            if not isinstance(raw_img_value, str) or not raw_img_value.strip():
                issues.append(
                    ValidationIssue(
                        field="metadata.arch_img_url",
                        message="Architecture image must be a non-empty base64 string.",
                        level="error",
                    )
                )
            else:
                base64_str = raw_img_value.strip()

                if base64_str.startswith("data:"):
                    comma_idx = base64_str.find(",")
                    if comma_idx != -1:
                        base64_str = base64_str[comma_idx + 1 :]

                try:
                    img_bytes = base64.b64decode(base64_str, validate=True)
                    if not img_bytes:
                        raise ValueError("Decoded image is empty")
                except Exception:
                    issues.append(
                        ValidationIssue(
                            field="metadata.arch_img_url",
                            message="arch_img_url is not valid base64-encoded image data.",
                            level="error",
                        )
                    )

            is_valid = not any(
                str(issue.level).lower() == "error" for issue in issues
            )

            result = InputValidationResult(
                is_valid=is_valid,
                issues=[asdict(issue) for issue in issues],
            )
            ctx.validation_result = result
            return result

        except Exception as e:
            logger.exception("[%s] InputValidationAgent failed", review_id)
            issues.append(
                ValidationIssue(
                    field="input_validation",
                    message=str(e),
                    level="error",
                )
            )
            result = InputValidationResult(is_valid=False, issues=[asdict(i) for i in issues])
            ctx.validation_result = result
            return result
