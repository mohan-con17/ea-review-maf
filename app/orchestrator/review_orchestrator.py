from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Any
from dataclasses import asdict

from app.logs import review_logger
from app.domain.review_models import ReviewSessionContext
from app.agents.input_validation_agent import InputValidationAgent
from app.agents.demographics_agent import DemographicsAgent
from app.agents.image_analyser_agent import ImageAnalyzerAgent
from app.agents.triage_agent import TriageAgent
from app.agents.formatting_agent import FormattingAgent
from app.agents.remidiation_agent import RemediationAgent
from app.orchestrator.planner_kernel import PlannerAgent, PlanDecision

class ReviewOrchestrator:

    def __init__(self) -> None:
        # Agents can be injected later
        self._validator = InputValidationAgent()
        self._planner = PlannerAgent()
        self._demographics_agent = DemographicsAgent()
        self._image_agent = ImageAnalyzerAgent()
        self._triage_agent = TriageAgent()
        self._remediation_agent = RemediationAgent()
        self._formatter = FormattingAgent()
        
        self._agents = {
            "demographics": DemographicsAgent(),
            "image_analysis": ImageAnalyzerAgent(),
            "triage": TriageAgent(),
            "remediation": RemediationAgent(),
        }
        
        self._logger = review_logger.AzureBlobLogger()

    async def review(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        review_id = metadata.get("REQUEST_NO", str(uuid.uuid4())) 

        ctx = ReviewSessionContext(review_id=review_id, metadata=metadata)
        
        try:
            # 1) Validate inputs
            validation_result = await self._validator.validate(ctx)

            if not validation_result.is_valid:
                return await self._formatter.format_failure_response(
                    ctx,
                    failure_details=getattr(validation_result, "issues", []),
                    failure_stage="input_validation"
                )

            # 2) Planner: decide staged layout of agents
            plan: PlanDecision
            try:
                plan = await self._planner.plan(ctx)
            except Exception as e:
                return await self._formatter.format_failure_response(
                    ctx,
                    failure_details=[{"level": "ERROR", "message": str(e)}],
                    failure_stage="planner"
                )

            # --- 3) Execute agent stages ---
            for stage in plan.stages:
                tasks = []
                for agent_name in stage:
                    agent = self._agents.get(agent_name)
                    if not agent:
                        continue
                    tasks.append(agent.run(ctx))
                if tasks:
                    await asyncio.gather(*tasks)

            # --- 4) Format final success response ---
            return await self._formatter.format_success_response(ctx, plan)

        finally:
            self._logger.log(review_id, asdict(ctx))