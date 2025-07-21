from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import structlog

logger = structlog.get_logger()

load_dotenv()


class Config:
    """Configuration management for the fundamental analysis system."""
    
    def __init__(self):
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.output_dir = Path(__file__).parent.parent.parent / "output"
        
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        logger.info("Configuration loaded", 
                   has_openai_key=bool(self.openai_api_key),
                   has_anthropic_key=bool(self.anthropic_api_key),
                   llm_model=self.llm_model)
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not (self.openai_api_key or self.anthropic_api_key):
            logger.error("No LLM API keys configured")
            return False
        return True


config = Config()