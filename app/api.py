from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.routers import screening

# Create FastAPI app
app = FastAPI(
    title="AI HR Resume Screener API",
    description="Production API for AI-powered candidate screening using CrewAI and Qdrant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include routers
app.include_router(screening.router, prefix="/api", tags=["screening"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
