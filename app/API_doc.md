# AI HR Resume Screener API Documentation

## Overview

The AI HR Resume Screener API provides a production-ready REST interface for screening and ranking job candidates using CrewAI multi-agent systems, Qdrant vector search, and LLM-powered evaluation.

**Base URL:** `http://localhost:8000`  
**API Version:** 1.0.0  
**Interactive Docs:** `/docs` (Swagger UI)  
**Alternative Docs:** `/redoc` (ReDoc)

---

## Authentication

All endpoints require API key authentication via the `X-API-Key` header.

### Headers
```
X-API-Key: your-api-key-here
```

### Authentication Error
```json
{
  "detail": "API key is required. Provide it in the X-API-Key header."
}
```

**Status Code:** `401 Unauthorized`


## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Screening endpoint:** 10 requests per hour per IP address
- **Other endpoints:** No rate limiting (subject to change in production)

### Rate Limit Exceeded
```json
{
  "detail": "Rate limit exceeded: 5 per 1 hour"
}
```

**Status Code:** `429 Too Many Requests`

---

## Endpoints

### 1. Root Endpoint

Get basic API information and available endpoints.

**Endpoint:** `GET /`  
**Rate Limit:** None

#### Response
```json
{
  "name": "AI HR Resume Screener API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/health",
  "endpoints": {
    "screen": "/api/screen",
    "professions": "/api/professions",
    "health": "/api/health",
    "job_result": "/api/jobs/{job_id}",
    "report": "/api/report/{report_id}"
  }
}
```

**Status Code:** `200 OK`

---

### 2. Health Check

Check the health status of Qdrant and LLM services.

**Endpoint:** `GET /api/health`  
**Rate Limit:** None

#### Response
```json
{
  "status": "healthy",
  "qdrant_status": "healthy",
  "llm_status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000000"
}
```

**Status Codes:**
- `200 OK` - All services healthy
- `503 Service Unavailable` - One or more services unhealthy

#### Status Values
- `status`: Overall health status (`healthy` or `unhealthy`)
- `qdrant_status`: Qdrant vector database status (`healthy` or `unhealthy`)
- `llm_status`: LLM service status (`healthy` or `unhealthy`)
- `timestamp`: ISO 8601 timestamp of the health check

---

### 3. Get Professions

Retrieve the list of available profession categories for candidate filtering.

**Endpoint:** `GET /api/professions`  
**Rate Limit:** None

#### Response
```json
{
  "professions": [
    "ENGINEERING",
    "FINANCE",
    "ACCOUNTING",
    "SALES",
    "MARKETING",
    "HR",
    "OPERATIONS",
    "LEGAL",
    "HEALTHCARE",
    "EDUCATION",
    "FITNESS",
    "ARTS",
    "CONSTRUCTION",
    "REAL_ESTATE",
    "RETAIL",
    "HOSPITALITY",
    "TRANSPORTATION",
    "AGRICULTURE",
    "MANUFACTURING",
    "TELECOMMUNICATIONS",
    "ENERGY",
    "MEDIA",
    "NON_PROFIT"
  ],
  "count": 24
}
```

**Status Code:** `200 OK`

---

### 4. Submit Screening Job

Submit a new candidate screening job. The screening process runs asynchronously in the background.

**Endpoint:** `POST /api/screen`  
**Rate Limit:** 5 requests per hour per IP

#### Request Body

```json
{
  "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience in data engineering, strong Python and SQL skills, and experience with cloud platforms like AWS or GCP.",
  "profession": "ENGINEERING",
  "top_k": 10
}
```

#### Request Schema

| Field | Type | Required | Constraints | Description |
|-------|------|----------|--------------|-------------|
| `job_description` | string | Yes | Min: 50 chars, Max: 3000 chars | Full job description text |
| `profession` | string | Yes | Must be a valid profession category | Profession category for filtering candidates |
| `top_k` | integer | No | Min: 3, Max: 20, Default: 10 | Number of candidates to retrieve |

#### Response (Job Submitted)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Screening job submitted successfully. Use the job_id to check status."
}
```

**Status Code:** `200 OK`

#### Response (Validation Error)

```json
{
  "detail": [
    {
      "loc": ["body", "job_description"],
      "msg": "ensure this value has at least 50 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

**Status Code:** `422 Unprocessable Entity`

#### Common Validation Errors
- Job description too short (< 50 characters)
- Job description too long (> 3000 characters)
- Invalid profession (not in the list of valid professions)
- Invalid top_k value (< 3 or > 20)

---

### 5. Get Job Result

Retrieve the result of a screening job by job ID. Poll this endpoint to check job status and retrieve results when complete.

**Endpoint:** `GET /api/jobs/{job_id}`  
**Rate Limit:** None

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string (UUID) | Yes | The job ID returned from the screening endpoint |

#### Response (Job Complete)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "report": {
    "job_title": "Senior Data Engineer",
    "profession": "ENGINEERING",
    "ranked_candidates": [
      {
        "rank": 1,
        "name": "John Doe",
        "overall_score": 8.5,
        "technical_fit": 9.0,
        "experience_level": 8.0,
        "culture_signals": 8.5,
        "red_flags": 9.0,
        "technical_skills_score": 9.0,
        "experience_score": 8.0,
        "education_score": 8.5,
        "strengths": [
          "Strong Python and SQL skills",
          "5+ years of data engineering experience",
          "Experience with AWS cloud platform"
        ],
        "gaps": [
          "Limited experience with real-time data processing",
          "No experience with dbt"
        ],
        "verdict": "STRONG FIT",
        "compensation_fit": "appropriate"
      }
    ],
    "top_recommendation": "John Doe - Strong technical fit with 5+ years of relevant experience and excellent cloud platform skills.",
    "summary": "The candidate pool shows strong technical skills overall. John Doe stands out as the top candidate with exceptional Python/SQL skills and relevant cloud experience. Two other candidates show promise but lack specific cloud platform experience.",
    "generated_at": "2024-01-01T12:00:00.000000",
    "total_candidates_evaluated": 10
  },
  "markdown_report": "# Hiring Report\n\n..."
}
```

**Status Code:** `200 OK`

#### Response (Job In Progress)

```json
{
  "detail": "Job status: processing. Please check again later."
}
```

**Status Code:** `202 Accepted`

#### Response (Job Failed)

```json
{
  "detail": "Screening job failed: Failed to parse JSON output from screening pipeline"
}
```

**Status Code:** `500 Internal Server Error`

#### Response (Job Not Found)

```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**Status Code:** `404 Not Found`

#### Job Status Values
- `pending` - Job queued, waiting to start
- `processing` - Job is currently running (typically takes 3-8 minutes)
- `complete` - Job finished successfully, results available
- `failed` - Job failed with an error

---

### 6. Get Report

Retrieve a previously generated screening report by report ID.

**Endpoint:** `GET /api/report/{report_id}`  
**Rate Limit:** None

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `report_id` | string (UUID) | Yes | The report ID (same as job_id for completed jobs) |

#### Response (Success)

```json
{
  "report": {
    "job_title": "Senior Data Engineer",
    "profession": "ENGINEERING",
    "ranked_candidates": [...],
    "top_recommendation": "...",
    "summary": "...",
    "generated_at": "2024-01-01T12:00:00.000000",
    "total_candidates_evaluated": 10
  },
  "markdown_report": "# Hiring Report\n\n...",
  "completed_at": "2024-01-01T12:08:00.000000"
}
```

**Status Code:** `200 OK`

#### Response (Not Found)

```json
{
  "detail": "Report 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**Status Code:** `404 Not Found`

---

## Data Models

### ScreeningRequest

```python
{
  "job_description": str (min: 50, max: 3000),
  "profession": str,
  "top_k": int (min: 3, max: 20, default: 10)
}
```

### ScreeningResponse

```python
{
  "job_id": str (UUID),
  "status": Literal["pending", "processing", "complete", "failed"],
  "message": str
}
```

### ScreeningResult

```python
{
  "job_id": str (UUID),
  "status": Literal["complete"],
  "report": HiringReport,
  "markdown_report": Optional[str]
}
```

### CandidateScorecard

```python
{
  "rank": int,
  "name": str,
  "overall_score": float (0-10),
  "technical_fit": float (0-10),
  "experience_level": float (0-10),
  "culture_signals": float (0-10),
  "red_flags": float (0-10),
  "technical_skills_score": float (0-10),
  "experience_score": float (0-10),
  "education_score": float (0-10),
  "strengths": List[str],
  "gaps": List[str],
  "verdict": Literal["STRONG FIT", "GOOD FIT", "PARTIAL FIT", "WEAK FIT"],
  "compensation_fit": Optional[str]
}
```

### HiringReport

```python
{
  "job_title": str,
  "profession": str,
  "ranked_candidates": List[CandidateScorecard],
  "top_recommendation": str,
  "summary": str,
  "generated_at": str (ISO 8601),
  "total_candidates_evaluated": int
}
```

### HealthResponse

```python
{
  "status": Literal["healthy", "unhealthy"],
  "qdrant_status": str,
  "llm_status": str,
  "timestamp": str (ISO 8601)
}
```

### ProfessionsResponse

```python
{
  "professions": List[str],
  "count": int
}
```

---

## Usage Examples

### Example 1: Complete Screening Workflow

#### Step 1: Submit Screening Job

```bash
curl -X POST http://localhost:8000/api/screen \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience in data engineering, strong Python and SQL skills, and experience with cloud platforms like AWS or GCP.",
    "profession": "ENGINEERING",
    "top_k": 10
  }'
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Screening job submitted successfully. Use the job_id to check status."
}
```

#### Step 2: Poll for Results

```bash
curl http://localhost:8000/api/jobs/550e8400-e29b-41d4-a716-446655440000
```

**Response (Still Processing):**
```json
{
  "detail": "Job status: processing. Please check again later."
}
```

**Status Code:** `202 Accepted`

#### Step 3: Retrieve Final Results

```bash
curl http://localhost:8000/api/jobs/550e8400-e29b-41d4-a716-446655440000
```

**Response (Complete):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "complete",
  "report": {
    "job_title": "Senior Data Engineer",
    "profession": "ENGINEERING",
    "ranked_candidates": [...],
    "top_recommendation": "...",
    "summary": "...",
    "generated_at": "2024-01-01T12:00:00.000000",
    "total_candidates_evaluated": 10
  },
  "markdown_report": "# Hiring Report\n\n..."
}
```

---

### Example 2: Check Available Professions

```bash
curl http://localhost:8000/api/professions
```

**Response:**
```json
{
  "professions": [
    "ENGINEERING",
    "FINANCE",
    "ACCOUNTING",
    ...
  ],
  "count": 24
}
```

---

### Example 3: Health Check

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "qdrant_status": "healthy",
  "llm_status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000000"
}
```

---

### Example 4: Python Client Example

```python
import requests
import time

API_BASE = "http://localhost:8000"

# Submit screening job
screening_request = {
    "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience...",
    "profession": "ENGINEERING",
    "top_k": 10
}

response = requests.post(
    f"{API_BASE}/api/screen",
    json=screening_request
)

if response.status_code == 200:
    job_id = response.json()["job_id"]
    print(f"Job submitted: {job_id}")
    
    # Poll for results
    while True:
        result_response = requests.get(
            f"{API_BASE}/api/jobs/{job_id}"
        )
        
        if result_response.status_code == 200:
            result = result_response.json()
            print(f"Job status: {result['status']}")
            
            if result["status"] == "complete":
                print("Screening complete!")
                print(f"Top recommendation: {result['report']['top_recommendation']}")
                break
            elif result["status"] == "failed":
                print(f"Job failed: {result_response.json()['detail']}")
                break
        
        time.sleep(30)  # Wait 30 seconds before polling again
```

---

## Error Handling

### Common HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `200 OK` | Request successful |
| `202 Accepted` | Request accepted, processing in background |
| `404 Not Found` | Resource not found (job_id, report_id) |
| `422 Unprocessable Entity` | Validation error in request body |
| `429 Too Many Requests` | Rate limit exceeded |
| `500 Internal Server Error` | Server error or job failure |
| `503 Service Unavailable` | Service unhealthy (Qdrant or LLM down) |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Architecture

### Screening Pipeline Flow

1. **Client submits screening job** → `POST /api/screen`
2. **Server validates input** → Checks job description length, profession validity, top_k range
3. **Server generates job_id** → UUID for tracking
4. **Background task starts** → CrewAI pipeline runs asynchronously
5. **Pipeline executes**:
   - Task 1: Job Analysis (extract requirements)
   - Task 2: Candidate Retrieval (from Qdrant)
   - Task 3: Candidate Evaluation (multi-dimension scoring)
   - Task 4: Report Generation (JSON + markdown)
6. **Results stored** → In-memory storage (production: Redis/DB)
7. **Client polls for results** → `GET /api/jobs/{job_id}`
8. **Results returned** → Structured JSON report + markdown

### Multi-Dimension Scoring

Each candidate is evaluated across four dimensions:

- **technical_fit (0-10):** Alignment with technical skills and tools required
- **experience_level (0-10):** Years and relevance of experience
- **culture_signals (0-10):** Soft skills, communication, leadership potential
- **red_flags (0-10):** Risk assessment (10 = no red flags, 0 = major concerns)

**Overall Score Calculation:**
```
overall_score = (technical_fit * 0.40) + 
                (experience_level * 0.30) + 
                (culture_signals * 0.20) + 
                (red_flags * 0.10)
```

---

## Running the API

### Development Mode

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Coming Soon)

```bash
docker-compose up
```

---

## Environment Variables

Required environment variables (set in `.env`):

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## Testing

Run the integration test suite:

```bash
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_api.py::test_health_check -v
```

---

## Rate Limiting Details

- **Endpoint:** `POST /api/screen`
- **Limit:** 5 requests per hour per IP address
- **Implementation:** slowapi with IP-based key function
- **Response on Exceed:** HTTP 429 with error message

---

## Responsible AI

Every screening report includes a "Limitations & Bias Considerations" section that:

- Flags that AI does not have access to candidate demographics
- States that AI cannot guarantee bias-free ranking
- Emphasizes that scores are recommendations, not decisions
- Requires human review before any offer or rejection
- Notes that scoring is based solely on resume text
- Clarifies that employment gaps or career changes are not automatically negative
- Positions the system as one input among many in a holistic hiring process

---

## Support

For issues, questions, or contributions, please refer to the project repository.

---

## Changelog

### Version 1.0.0 (2024-01-01)
- Initial release
- FastAPI wrapper around CrewAI pipeline
- Authentication via API key
- Rate limiting (5 requests/hour)
- Input validation
- Multi-dimension candidate scoring
- Responsible AI disclaimers
- Integration tests
- CI/CD with GitHub Actions
