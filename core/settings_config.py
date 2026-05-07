from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "hr_resumes"
    top_k: int = 10

    delay_between_requests: int = 2
    max_retries: int = 3
    retry_wait: int = 10
    progress_file: str = "ingestion_progress.json"
    embedding_model: str = "gemini-embedding-001"

    gemini_api_key: str
    groq_api_key: str
    exa_api_key: str
    openrouter_api_key: str
    cohere_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()