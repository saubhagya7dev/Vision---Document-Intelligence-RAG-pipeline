from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "vision_rag_docs"
    qdrant_in_memory: bool = True  # Set to False when using a real Qdrant server (e.g. Docker)
    
    # Models
    embedding_model_name: str = "vidore/colpali"
    generation_model_name: str = "gemini-1.5-pro"
    
    # API Keys
    google_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Global settings instance
settings = Settings()
