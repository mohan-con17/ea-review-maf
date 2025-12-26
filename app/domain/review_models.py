from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

@dataclass
class AgentScore:
    accuracy: float
    bias: float
    hallucination: float
    confidence: float
    notes: Dict[str, str]

@dataclass
class ReviewScores:
    per_agent: Dict[str, AgentScore] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    status: str = "success"   # success | failed
    error: str | None = None

class ReviewDecision(str, Enum):
    PROCEED = "proceed"
    REJECT = "reject"
    NEED_MORE_INFO = "need_more_info"

@dataclass
class ValidationIssue:
    field: str
    message: str
    level: str  # "error" | "warning"

@dataclass
class InputValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]

@dataclass
class PreprocessedImage:
    content_type: str                 # e.g. "image/png"
    ext: str                          # e.g. "png", "jpg"
    width: int
    height: int
    tiles: List[bytes]                # raw PNG/JPEG bytes per tile
    tiles_x: int
    tiles_y: int

@dataclass
class ArchitectureContext:
    system_name: str
    business_unit: str
    region: str
    criticality: str
    data_sensitivity: str
    compliance_tags: List[str]
    assumptions: List[str]

@dataclass
class ComponentNode:
    id: str
    name: str
    type: str
    technology: Optional[str] = None

@dataclass
class ConnectionEdge:
    from_id: str
    to_id: str
    description: str
    
class PlanDecision(BaseModel):
    stages: List[List[str]]
    notes: Optional[str] = None

    def all_agents(self) -> List[str]:
        """Flatten stages into a simple list of agent names."""
        return [agent for stage in self.stages for agent in stage]

@dataclass
class DiagramAnalysis:
    components: List[ComponentNode]
    connections: List[ConnectionEdge]
    risks: List[str]
    observations: List[str]

@dataclass
class RemediationSnapshot:
    # From the chosen template document
    combination_path: Optional[str] = None
    template: Dict[str, Any] = field(default_factory=dict)
    # Comparison results
    missing_components_in_triage: List[str] = field(default_factory=list)
    available_components_in_triage: List[str] = field(default_factory=list)
    missing_components_in_template: List[str] = field(default_factory=list)
    similarity_percent: List[str] = field(default_factory=list)

@dataclass
class ReviewResult:
    review_id: str
    validation: InputValidationResult
    context: Optional[ArchitectureContext]
    diagram: Optional[DiagramAnalysis]
    recommendations: List[RemediationSnapshot]
    
@dataclass
class DemographicsResult:
    users: Optional[str] = None           # e.g., "Internal", "External", "Mixed"
    sub_users: Optional[str] = None       # e.g., "Branch", "User", "Partner"
    network: Optional[str] = None         # e.g., "Internet - VPN", "MPLS", "Internet - VDI"
    deployment: Optional[str] = None      # e.g., "cloud", "on prem"
    cloud_provider: Optional[str] = None  # e.g., "Azure", "AWS", "OCI", "NA"
    tier: Optional[str] = None            # e.g., "1", "2", "3", "4", "5"

@dataclass
class ImageAnalysisResult:
    architecture_summary: Optional[str] = None
    image_components_json: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FormatterResult:
    review_summary: Optional[str] = None

@dataclass
class ReviewSessionContext:
    """
    Single object that carries state from start to end of a review.
    Orchestrator creates it once and passes it to agents, which can
    update relevant fields.
    """
    review_id: str
    metadata: Dict[str, Any]
    computed_metadata: Dict[str, Any] = field(default_factory=dict)
    files: List[Any] = field(default_factory=list)

    # populated as pipeline runs
    validation_result: Optional[InputValidationResult] = None
    
    # NEW: non-LLM pre-processing output
    preprocessed_image: Optional[PreprocessedImage] = None

    demographics_from_json: Optional[DemographicsResult] = None
    image_analysis: Optional[ImageAnalysisResult] = None
    triage_results: Dict[str, Any] = field(default_factory=dict)
    remediation_result: Optional[RemediationSnapshot] = None
    formatting_summary: Optional[FormatterResult] = None
    
    # per-agent SLA timing (MAF telemetry)
    agent_sla: List[Dict[str, Any]] = field(default_factory=list)
    llm_traces: List[Dict[str, Any]] = field(default_factory=list)
    review_scores: ReviewScores | None = None
