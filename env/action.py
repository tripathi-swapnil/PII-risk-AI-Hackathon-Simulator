from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ActionType(str, Enum):
    DETECT = "detect"
    CLASSIFY = "classify"
    REDACT = "redact"
    ESCALATE = "escalate"
    FINALIZE = "finalize"

class Span(BaseModel): 
    start: int
    end: int
    label: str

class Action(BaseModel):
    action_type: ActionType
    spans: Optional[List[Span]] = None
    classification: Optional[str] = None
    redacted_text: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None