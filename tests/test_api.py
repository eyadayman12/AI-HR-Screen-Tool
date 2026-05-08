import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock
from app.api import app
from app.routers.screening import jobs, job_reports


@pytest.fixture
def client():
    """Fixture for AsyncClient."""
    return AsyncClient(app=app, base_url="http://test")


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test the root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data


@pytest.mark.asyncio
async def test_health_check(client):
    """Test the health check endpoint."""
    with patch('app.routers.screening.QdrantClient') as mock_qdrant:
        mock_qdrant_instance = Mock()
        mock_qdrant_instance.get_collections.return_value = []
        mock_qdrant.return_value = mock_qdrant_instance
        
        with patch('app.routers.screening.gemini_llm', Mock()):
            response = await client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "qdrant_status" in data
            assert "llm_status" in data


@pytest.mark.asyncio
async def test_get_professions(client):
    """Test the professions endpoint."""
    response = await client.get("/api/professions")
    assert response.status_code == 200
    data = response.json()
    assert "professions" in data
    assert "count" in data
    assert isinstance(data["professions"], list)
    assert len(data["professions"]) > 0


@pytest.mark.asyncio
async def test_screen_candidates_success(client):
    """Test successful screening job submission."""
    request_data = {
        "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience in data engineering, strong Python and SQL skills, and experience with cloud platforms like AWS or GCP.",
        "profession": "ENGINEERING",
        "top_k": 5
    }
    
    # Mock the background task
    with patch('app.routers.screening.BackgroundTasks.add_task') as mock_add_task:
        response = await client.post("/api/screen", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"
        assert mock_add_task.called


@pytest.mark.asyncio
async def test_screen_candidates_invalid_profession(client):
    """Test screening with invalid profession returns 422."""
    request_data = {
        "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience in data engineering, strong Python and SQL skills, and experience with cloud platforms like AWS or GCP.",
        "profession": "INVALID_PROFESSION",
        "top_k": 5
    }
    
    response = await client.post("/api/screen", json=request_data)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_screen_candidates_short_description(client):
    """Test screening with job description under 50 characters returns 422."""
    request_data = {
        "job_description": "Too short",
        "profession": "ENGINEERING",
        "top_k": 5
    }
    
    response = await client.post("/api/screen", json=request_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_screen_candidates_long_description(client):
    """Test screening with job description over 3000 characters returns 422."""
    long_description = "A" * 3001
    request_data = {
        "job_description": long_description,
        "profession": "ENGINEERING",
        "top_k": 5
    }
    
    response = await client.post("/api/screen", json=request_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_screen_candidates_invalid_top_k(client):
    """Test screening with invalid top_k (under 3) returns 422."""
    request_data = {
        "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience in data engineering, strong Python and SQL skills, and experience with cloud platforms like AWS or GCP.",
        "profession": "ENGINEERING",
        "top_k": 2
    }
    
    response = await client.post("/api/screen", json=request_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_job_result_not_found(client):
    """Test getting a non-existent job returns 404."""
    response = await client.get("/api/jobs/non-existent-job-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_report_not_found(client):
    """Test getting a non-existent report returns 404."""
    response = await client.get("/api/report/non-existent-report-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_result_complete(client):
    """Test getting a completed job result."""
    # Setup: Create a completed job
    job_id = "test-job-id"
    jobs[job_id] = {
        "status": "complete",
        "created_at": "2024-01-01T00:00:00",
        "request": {}
    }
    job_reports[job_id] = {
        "report": {
            "job_title": "Test Job",
            "profession": "ENGINEERING",
            "ranked_candidates": [],
            "top_recommendation": "Test Candidate",
            "summary": "Test summary",
            "generated_at": "2024-01-01T00:00:00",
            "total_candidates_evaluated": 0
        },
        "markdown_report": "# Test Report",
        "completed_at": "2024-01-01T00:00:00"
    }
    
    response = await client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert "report" in data
    
    # Cleanup
    del jobs[job_id]
    del job_reports[job_id]


@pytest.mark.asyncio
async def test_get_report_success(client):
    """Test getting an existing report."""
    # Setup: Create a report
    report_id = "test-report-id"
    job_reports[report_id] = {
        "report": {
            "job_title": "Test Job",
            "profession": "ENGINEERING",
            "ranked_candidates": [],
            "top_recommendation": "Test Candidate",
            "summary": "Test summary",
            "generated_at": "2024-01-01T00:00:00",
            "total_candidates_evaluated": 0
        },
        "markdown_report": "# Test Report",
        "completed_at": "2024-01-01T00:00:00"
    }
    
    response = await client.get(f"/api/report/{report_id}")
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "markdown_report" in data
    
    # Cleanup
    del job_reports[report_id]