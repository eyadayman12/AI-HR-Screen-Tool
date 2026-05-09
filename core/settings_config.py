from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "hr_resumes"
    top_k: int = 50
    qdrant_timeout: int = 60
    rank_top_k: int = 10

    delay_between_requests: int = 2
    max_retries: int = 3
    retry_wait: int = 10
    progress_file: str = "ingestion_progress.json"
    embedding_model: str = "gemini-embedding-001"
    reranker_model: str = "rerank-v4.0-pro"
    
    openrouter_model:str = "openrouter/openrouter/hunter-alpha"
    openrouter_base_url:str = "https://openrouter.ai/api/v1"
    cohere_model:str = "cohere/command-a-03-2025"
    gemini_model:str = "gemini/gemma-4-31b-it"
    groq_model:str =  "groq/llama-3.3-70b-versatile"
    model_temperature:float = 0.2

    gemini_api_key: str
    groq_api_key: str
    exa_api_key: str
    openrouter_api_key: str
    cohere_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()