# app/models/api_responses.py
from typing import List, Optional
from pydantic import BaseModel

class ValidationIssueDTO(BaseModel):
    field: str
    message: str
    level: str


class InputValidationResultDTO(BaseModel):
    is_valid: bool
    issues: List[ValidationIssueDTO]


class TriageDecisionDTO(BaseModel):
    decision: str
    message: str
    blocking_issues: List[str]


class ScoreResultDTO(BaseModel):
    security: int
    reliability: int
    performance: int
    cost_efficiency: int
    maintainability: int
    compliance: int
    summary: str


class ReviewSummaryResponse(BaseModel):
    """Structured result that can be fetched after streaming finishes."""
    review_id: str
    validation: InputValidationResultDTO
    triage: TriageDecisionDTO
    scores: Optional[ScoreResultDTO] = None
    # you can add more fields from domain as needed


class ReviewInitResponse(BaseModel):
    """If you choose a two-step pattern (start + fetch result)."""
    review_id: str
    message: str
