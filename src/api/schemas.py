from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 20
    context: Optional[dict] = None

class RecommendedItem(BaseModel):
    item_id: str
    score: float
    rank: int

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendedItem]
    model_version: str
    strategy: str
