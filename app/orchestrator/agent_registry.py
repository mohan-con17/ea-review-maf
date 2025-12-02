from __future__ import annotations
from typing import List, Dict, Any

_AGENTS: List[Dict[str, Any]] = [
    {
        "name": "demographics",
        "description": 
            "Uses Project Specifics / Questionnaires sections from metadata to infer tier, deployment/hosting, primary user types, and access modes.",
        "dependencies": [],
    },
    {
        "name": "image_analysis",
        "description": 
            "Uses the arch_img_url section from metadata which stores architecture diagram image in base64 format to infer topology, hosting/exposure, notable components and diagram-based risks.",
        "dependencies": [],
    },
    {
        "name": "triage",
        "description": 
            "Uses outputs from demographics and image_analysis to prioritize findings and prepare context for remediation.",
        "dependencies": ["demographics", "image_analysis"],
    },
    {
        "name": "remediation",
        "description": 
            "Uses outputs from triage to select a JSON stored in Azure Blob using Azure AI Search based on tier, hosting, user type, and access modes.",
        "dependencies": ["triage"],
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_agents_definition() -> List[Dict[str, Any]]:

    # Return a deep copy to prevent accidental mutation by callers
    import copy
    return copy.deepcopy(_AGENTS)
