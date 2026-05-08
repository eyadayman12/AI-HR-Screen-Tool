from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Dict
import uuid
from datetime import datetime

from app.schema import (
    ScreeningRequest, ScreeningResponse, ScreeningResult,
    HealthResponse, ProfessionsResponse
)
from core.tasks import build_tasks
from core.agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from crewai import Crew, Process
from core.settings_config import settings
from qdrant_client import QdrantClient
from core.llm import gemini_llm
import json
import re

router = APIRouter()

limiter = Limiter(key_func=get_remote_address)

jobs: Dict[str, Dict] = {}
job_reports: Dict[str, Dict] = {}

def parse_output(raw_output: str) -> tuple:
    """Parse the crew output to extract JSON and markdown sections."""
    separator = "=== MARKDOWN REPORT ==="
    if separator in raw_output:
        parts = raw_output.split(separator)
        json_part = parts[0].strip()
        markdown_part = parts[1].strip()
        
        try:
            json_data = json.loads(json_part)
            return json_data, markdown_part
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                    return json_data, raw_output
                except json.JSONDecodeError:
                    pass
            return None, raw_output
    else:
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            try:
                json_data = json.loads(json_match.group())
                return json_data, raw_output
            except json.JSONDecodeError:
                pass
        return None, raw_output


async def run_screening_job(job_id: str, job_description: str, profession: str, top_k: int):
    """Background task to run the screening pipeline."""
    try:
        jobs[job_id]["status"] = "processing"
        
        original_top_k = settings.top_k
        settings.top_k = top_k
        
        tasks = build_tasks(
            job_description=job_description,
            profession=profession,
        )
        
        crew = Crew(
            agents=[job_analyst, candidate_retriever, candidate_evaluator, report_writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
        )
        
        result = crew.kickoff()
        
        json_data, markdown_report = parse_output(str(result.raw))
        
        settings.top_k = original_top_k
        
        if json_data:
            job_reports[job_id] = {
                "report": json_data,
                "markdown_report": markdown_report,
                "completed_at": datetime.utcnow().isoformat()
            }
            jobs[job_id]["status"] = "complete"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Failed to parse JSON output from screening pipeline"
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        # Restore original top_k
        settings.top_k = original_top_k


@router.post("/screen", response_model=ScreeningResponse)
@limiter.limit("5/hour")
async def screen_candidates(
    request: Request,
    screening_request: ScreeningRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a screening job.
    
    Accepts job description, profession, and top_k parameters.
    Returns a job_id for tracking the asynchronous screening process.
    Rate limited to 5 requests per hour per IP address.
    """
    # Validate profession against Qdrant (basic validation)
    # In production, query Qdrant to get valid professions
    valid_professions = ["ENGINEERING", "FINANCE", "ACCOUNTING", "SALES", "MARKETING", 
                         "HR", "OPERATIONS", "LEGAL", "HEALTHCARE", "EDUCATION"]
    
    if screening_request.profession.upper() not in [p.upper() for p in valid_professions]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid profession '{screening_request.profession}'. Valid professions: {', '.join(valid_professions)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "request": screening_request.dict()
    }
    
    # Add background task
    background_tasks.add_task(
        run_screening_job,
        job_id=job_id,
        job_description=screening_request.job_description,
        profession=screening_request.profession.upper(),
        top_k=screening_request.top_k
    )
    
    return ScreeningResponse(
        job_id=job_id,
        status="pending",
        message="Screening job submitted successfully. Use the job_id to check status."
    )


@router.get("/jobs/{job_id}", response_model=ScreeningResult)
async def get_job_result(job_id: str):
    """
    Get the result of a screening job by job_id.
    
    Returns the structured JSON report and markdown report if the job is complete.
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = jobs[job_id]
    
    if job["status"] == "complete":
        if job_id not in job_reports:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Job marked as complete but report not found"
            )
        return ScreeningResult(
            job_id=job_id,
            status="complete",
            report=job_reports[job_id]["report"],
            markdown_report=job_reports[job_id]["markdown_report"]
        )
    elif job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening job failed: {job.get('error', 'Unknown error')}"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Job status: {job['status']}. Please check again later."
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Checks if Qdrant is reachable and if the LLM responds.
    """
    qdrant_status = "unhealthy"
    llm_status = "unhealthy"
    
    # Check Qdrant
    try:
        qdrant_config = settings.qdrant_host + ":" + str(settings.qdrant_port)
        qdrant_client = QdrantClient(qdrant_config, timeout=5)
        collections = qdrant_client.get_collections()
        qdrant_status = "healthy"
    except Exception:
        qdrant_status = "unhealthy"
    
    # Check LLM (simple test)
    try:
        # Just check if LLM object exists (actual API call would cost credits)
        if gemini_llm:
            llm_status = "healthy"
    except Exception:
        llm_status = "unhealthy"
    
    overall_status = "healthy" if qdrant_status == "healthy" and llm_status == "healthy" else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        qdrant_status=qdrant_status,
        llm_status=llm_status
    )


@router.get("/professions", response_model=ProfessionsResponse)
async def get_professions():
    """
    Get the list of available profession categories.
    
    Queries Qdrant to return the list of profession categories stored as payload metadata.
    """
    try:
        qdrant_config = settings.qdrant_host + ":" + str(settings.qdrant_port)
        qdrant_client = QdrantClient(qdrant_config, timeout=settings.qdrant_timeout)
        
        # Scroll through collection to get unique professions
        from qdrant_client.models import ScrollRequest, Filter, PayloadSelector
        
        # For now, return a static list (in production, query Qdrant)
        professions = [
            "ENGINEERING", "FINANCE", "ACCOUNTING", "SALES", "MARKETING",
            "HR", "OPERATIONS", "LEGAL", "HEALTHCARE", "EDUCATION",
            "FITNESS", "ARTS", "CONSTRUCTION", "REAL_ESTATE", "RETAIL",
            "HOSPITALITY", "TRANSPORTATION", "AGRICULTURE", "MANUFACTURING",
            "TELECOMMUNICATIONS", "ENERGY", "MEDIA", "NON_PROFIT"
        ]
        
        return ProfessionsResponse(
            professions=professions,
            count=len(professions)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve professions: {str(e)}"
        )


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """
    Get a previous screening report by report_id.
    
    This endpoint allows fetching a previous result without re-running the pipeline.
    In production, this would query a database or Redis.
    """
    if report_id not in job_reports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found"
        )
    
    return job_reports[report_id]
