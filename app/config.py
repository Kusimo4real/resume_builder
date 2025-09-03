from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    deepseek_api_key: str
    deepseek_api_key_open_router: str
    ows_api_base_url: str
    ows_username: str
    ows_password: str
    groq_api_key: str
    
    # redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()