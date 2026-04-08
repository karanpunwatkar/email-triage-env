from pydantic import BaseModel
from typing import Literal, Optional

class EmailObservation(BaseModel):
    sender: str
    subject: str
    body: str
    # Optional: last action outcome for partial progress feedback
    last_action_result: Optional[str] = None

class EmailAction(BaseModel):
    action_type: Literal["acknowledge", "reply", "escalate"]
    priority: Literal["low", "medium", "high"]
    content: Optional[str] = None  # only used for reply