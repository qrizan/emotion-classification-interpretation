# konfigurasi aplikasi - semua default values di satu tempat.
# bisa di-override via environment variables.

import os
from typing import Optional

# konfigurasi aplikasi dengan default values yang jelas.
class Config:
    
    # model configuration
    MODEL_ID: str = "qrizan/emotion-classifier-indonesia"
    MODEL_PATH: str = "qrizan/emotion-classifier-indonesia"
    
    # embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    
    # processing Configuration
    MAX_LENGTH: int = 128
    CONFIDENCE_THRESHOLD: float = 0.60
    
    # huggingFace API Configuration
    HF_TOKEN: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # download configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 10  # seconds
    
    # rate limiting configuration
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 30  # requests per minute per IP
    RATE_LIMIT_PER_HOUR: int = 200   # requests per hour per IP
    
    # get model path dari env atau default
    @classmethod
    def get_model_path(cls) -> str:
        return os.getenv("MODEL_PATH", cls.MODEL_PATH)
    
    # get model ID dari env atau default
    @classmethod
    def get_model_id(cls) -> str:
        return os.getenv("MODEL_ID", cls.MODEL_ID)
    
    # get embedding model name dari env atau default.
    @classmethod
    def get_embedding_model_name(cls) -> str:
        return os.getenv("EMBEDDING_MODEL_NAME", cls.EMBEDDING_MODEL_NAME)
    
    # get max length dari env atau default
    @classmethod
    def get_max_length(cls) -> int:
        return int(os.getenv("MAX_LENGTH", str(cls.MAX_LENGTH)))
    
    # get confidence threshold dari env atau default
    @classmethod
    def get_confidence_threshold(cls) -> float:
        return float(os.getenv("CONFIDENCE_THRESHOLD", str(cls.CONFIDENCE_THRESHOLD)))
    
    # get HF token dari env atau default
    @classmethod
    def get_hf_token(cls) -> Optional[str]:
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or cls.HF_TOKEN
    
    # get max retries dari env atau default
    @classmethod
    def get_max_retries(cls) -> int:
        return int(os.getenv("MAX_RETRIES", str(cls.MAX_RETRIES)))
    
    # get retry delay dari env atau default
    @classmethod
    def get_retry_delay(cls) -> int:
        return int(os.getenv("RETRY_DELAY", str(cls.RETRY_DELAY)))
    
    # get rate limit enabled dari env atau default
    @classmethod
    def get_rate_limit_enabled(cls) -> bool:
        return os.getenv("RATE_LIMIT_ENABLED", str(cls.RATE_LIMIT_ENABLED)).lower() == "true"
    
    # get rate limit per minute dari env atau default
    @classmethod
    def get_rate_limit_per_minute(cls) -> int:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE", str(cls.RATE_LIMIT_PER_MINUTE)))
    
    # get rate limit per hour dari env atau default
    @classmethod
    def get_rate_limit_per_hour(cls) -> int:
        return int(os.getenv("RATE_LIMIT_PER_HOUR", str(cls.RATE_LIMIT_PER_HOUR)))

