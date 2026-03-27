from pydantic import BaseModel

class JobRequest(BaseModel):
    description: str

class JobResponse(BaseModel):
    label: str
    confidence: float