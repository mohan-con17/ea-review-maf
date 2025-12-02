from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ArchitectureReviewRequest(BaseModel):
    """Incoming payload from frontend / client."""
    metadata: Dict[str, Any] = Field(
        ...,
        description="Raw system metadata JSON; will be validated and normalized."
    )
   # later add one more field for supporting documents based on user inputs