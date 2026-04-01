from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskType(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Entity(BaseModel):

    text: str
    label: str
    start: int
    end: int
    confidence: float = Field(..., ge=0.0, le=1.0)

class Observation(BaseModel):
    
    document_text: str
    detected_entities: List[Entity] = []
    risk_level: RiskLevel = RiskLevel.LOW
    task_type: TaskType
    step_count: int = 0
    feedback: str = ""
    done: bool = False
    constraint_violated: bool = False