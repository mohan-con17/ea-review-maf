from __future__ import annotations

import uuid
import time
from typing import Callable, Awaitable, Optional, Dict, Any
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

from app.logs import review_logger
from app.domain.review_models import ReviewSessionContext
from app.agents.input_validation_agent import InputValidationAgent
from app.agents.image_preprocessor import ImagePreprocessor
from app.agents.demographics_agent import DemographicsAgent
from app.agents.image_analyser_agent import ImageAnalyzerAgent
from app.agents.triage_agent import TriageAgent
from app.agents.formatting_agent import FormattingAgent
from app.agents.remidiation_agent import RemediationAgent
from app.agents.scoring_agent import ScoringAgent
from app.prompts.prompt_registry import PromptRegistry

ProgressCallback = Callable[[str, str, dict], Awaitable[None]]


AGENT_WEIGHTS = {
    "demographics": 0.1,
    "image_analysis": 0.4,
    "remediation": 0.4,
    "formatting": 0.1,
}


class ReviewOrchestrator:
    """
    TRUE MAF-style orchestrator:
    - Deterministic flow
    - Hard-fail semantics
    - Scoring before FE response
    - Logging after FE response
    """

    def __init__(self) -> None:
        PromptRegistry.load()

        self._validator = InputValidationAgent()
        self._image_preprocessor = ImagePreprocessor()
        self._demographics = DemographicsAgent()
        self._image_analyzer = ImageAnalyzerAgent()
        self._triage = TriageAgent()
        self._remediation = RemediationAgent()
        self._formatter = FormattingAgent()

        self._logger = review_logger.AzureBlobLogger()

    # ==============================================================
    # PUBLIC ENTRY POINT
    # ==============================================================

    async def review(
        self,
        metadata: Dict[str, Any],
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:

        review_id = metadata.get("REQUEST_NO", str(uuid.uuid4()))
        ctx = ReviewSessionContext(review_id=review_id, metadata=metadata)

        async def emit(stage: str, status: str, payload: dict | None = None):
            if progress_cb:
                await progress_cb(stage, status, payload or {})

        try:
            # ---------------- INPUT VALIDATION ----------------
            await emit("input_validation", "started")
            validation = await self._run_with_sla(
                ctx, "input_validation", self._validator.validate
            )
            await emit("Validating_user_inputs", "completed", {"is_valid": validation.is_valid})

            if not validation.is_valid:
                payload = await self._fail(
                    ctx, "input_validation", validation.issues, emit
                )
                self._log(ctx)
                return payload

            # ---------------- IMAGE PREPROCESS ----------------
            await emit("image_being_preprocessed", "started")
            await self._run_with_sla(ctx, "image_preprocessing", self._image_preprocessor.run)
            await emit("image_preprocessed", "completed")

            # ---------------- DEMOGRAPHICS ----------------
            await emit("Parsing_Architecture_Specific_details", "started")
            await self._run_with_sla(ctx, "demographics", self._demographics.run)
            await emit("Parsed_Architecture_details", "completed")

            # ---------------- IMAGE ANALYSIS ----------------
            await emit("image_analysis", "started")
            await self._run_with_sla(ctx, "image_analysis", self._image_analyzer.run)
            await emit("image_analysis", "completed")

            # ---------------- TRIAGE ----------------
            await self._run_with_sla(ctx, "triage", self._triage.run)

            # ---------------- REMEDIATION ----------------
            await emit("Comparing_Current_Architecture_&_Standard_Practices", "started")
            await self._run_with_sla(ctx, "remediation", self._remediation.run)
            await emit("Comparison", "completed")

            # ---------------- FORMAT ----------------
            await emit("formatting_user_response", "started")
            payload = await self._run_with_sla(
                ctx, "formatting", self._formatter.format_success_response
            )
            await emit("Response_is_almost_ready", "completed", payload)

            # ---------------- SCORING ----------------
            await self._run_scoring(ctx, emit)

            # ---------------- FINAL LOGGING ----------------
            self._log(ctx)
            return payload

        except Exception as e:
            payload = await self._fail(ctx, "orchestration", [str(e)], emit)
            self._log(ctx)
            return payload

    # ==============================================================
    # CORE UTILITIES
    # ==============================================================

    async def _run_with_sla(self, ctx, agent_name, fn):
        ctx.last_input_tokens = None
        ctx.last_output_tokens = None
        ctx.last_total_tokens = None

        start_ts = time.time()
        start_iso = datetime.now(timezone.utc).isoformat()
        status = "success"

        try:
            result = await fn(ctx)
            if hasattr(result, "error") and result.error:
                raise RuntimeError(result.error)
            return result
        except Exception:
            status = "failed"
            raise
        finally:
            end_ts = time.time()
            ctx.agent_sla.append({
                "agent": agent_name,
                "start_time": start_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_ms": int((end_ts - start_ts) * 1000),
                "duration_sec": round((end_ts - start_ts), 2),
                "input_tokens": ctx.last_input_tokens,
                "output_tokens": ctx.last_output_tokens,
                "total_tokens": ctx.last_total_tokens,
                "status": status,
            })

    async def _fail(self, ctx, stage, issues, emit):
        failure_details = [
            asdict(i) if is_dataclass(i) else {"message": str(i)}
            for i in issues
        ]

        payload = await self._formatter.format_failure_response(
            ctx, failure_details, stage
        )
        await emit("formatting", "completed_failure", payload)
        return payload

    # ==============================================================
    # SCORING
    # ==============================================================

    async def _run_scoring(self, ctx: ReviewSessionContext, emit):
        scorer = ScoringAgent()

        await emit("Evaluating_the_Responses", "started", {"message": "Scoring LLM responses"})

        try:
            per_agent = {}
            dimension_totals = {
                "accuracy": 0.0,
                "bias": 0.0,
                "hallucination": 0.0,
                "confidence": 0.0,
            }

            weight_total = 0.0

            for trace in ctx.llm_traces:
                agent = trace["agent"]
                if agent not in AGENT_WEIGHTS:
                    continue

                score = await scorer.score(trace)

                per_agent[agent] = {
                    **vars(score),
                    "weight": AGENT_WEIGHTS[agent],
                }

                for dim in dimension_totals:
                    dimension_totals[dim] += getattr(score, dim) * AGENT_WEIGHTS[agent]

                weight_total += AGENT_WEIGHTS[agent]

                await emit("Evaluating_agent_responses", f"{agent}_completed", {"agent": agent})

            # Normalize per-dimension scores
            overall_dimensions = {
                k: round(v / weight_total, 1)
                for k, v in dimension_totals.items()
            }

            # ---------------------------------------------
            # FINAL OVERALL SCORE
            # ---------------------------------------------
            overall_score = round(
                sum(overall_dimensions.values()) / len(overall_dimensions),
                1,
            )

            ctx.review_scores = {
                "status": "success",
                "scale": "1-10",
                "per_agent": per_agent,
                "overall": {
                    "accuracy": overall_dimensions["accuracy"],
                    "bias": overall_dimensions["bias"],
                    "hallucination": overall_dimensions["hallucination"],
                    "confidence": overall_dimensions["confidence"],
                    "score": overall_score,
                    "weights": AGENT_WEIGHTS,
                },
            }

            await emit("evaluating_agent_responses", "completed", ctx.review_scores)

        except Exception as e:
            ctx.review_scores = {"status": "failed", "error": str(e)}
            await emit("scoring", "failed", ctx.review_scores)

    def _log(self, ctx: ReviewSessionContext):
        try:
            self._logger.log(ctx.review_id, asdict(ctx))
        except Exception as e:
            print("Logging failed:", e)
