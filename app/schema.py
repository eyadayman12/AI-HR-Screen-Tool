from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime


class CandidateScorecard(BaseModel):
    """Individual candidate evaluation result with multi-dimension scoring."""
    rank: int
    name: str
    overall_score: float = Field(..., ge=0, le=10, description="Weighted average of all dimension scores")
    technical_fit: float = Field(..., ge=0, le=10, description="Technical skills match score")
    experience_level: float = Field(..., ge=0, le=10, description="Experience match score")
    culture_signals: float = Field(..., ge=0, le=10, description="Culture and soft skills alignment")
    red_flags: float = Field(..., ge=0, le=10, description="Risk assessment (lower is better, inverted for scoring)")
    technical_skills_score: float = Field(..., ge=0, le=10)
    experience_score: float = Field(..., ge=0, le=10)
    education_score: float = Field(..., ge=0, le=10)
    strengths: List[str]
    gaps: List[str]
    verdict: str
    compensation_fit: Optional[str] = None


class HiringReport(BaseModel):
    """Structured output from the screening pipeline."""
    job_title: str
    profession: str
    ranked_candidates: List[CandidateScorecard]
    top_recommendation: str
    summary: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    total_candidates_evaluated: int


# API Request/Response Schemas
class ScreeningRequest(BaseModel):
    """Request schema for the screening endpoint."""
    job_description: str = Field(..., min_length=50, max_length=3000, description="Job description text")
    profession: str = Field(..., description="Profession category (e.g., ENGINEERING, FINANCE)")
    top_k: int = Field(default=10, ge=3, le=20, description="Number of candidates to retrieve")


class ScreeningResponse(BaseModel):
    """Response schema for the screening endpoint."""
    job_id: str
    status: Literal["pending", "processing", "complete", "failed"]
    message: str


class ScreeningResult(BaseModel):
    """Response schema for completed screening results."""
    job_id: str
    status: Literal["complete"]
    report: HiringReport
    markdown_report: Optional[str] = None


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""
    status: Literal["healthy", "unhealthy"]
    qdrant_status: str
    llm_status: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ProfessionsResponse(BaseModel):
    """Response schema for the professions endpoint."""
    professions: List[str]
    count: int
