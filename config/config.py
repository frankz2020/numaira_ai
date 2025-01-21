from typing import Dict
import os
from pathlib import Path
from .base_config import BaseConfig

class Config(BaseConfig):
    """Environment-specific configuration that inherits from BaseConfig.
    Override values here for different environments (dev, prod, test)."""
    
    def __init__(self, env: str = "dev"):
        self.env = env
        self._load_env_specific_config()
    
    def _load_env_specific_config(self):
        """Load environment specific configurations"""
        if self.env == "prod":
            # Override production specific settings
            self.MODEL_CONFIG["llm"]["timeout"] = 60
            self.MODEL_CONFIG["embedding"]["similarity_threshold"] = 0.5
        elif self.env == "test":
            # Override test specific settings
            self.MODEL_CONFIG["llm"]["timeout"] = 10
            self.MODEL_CONFIG["embedding"]["similarity_threshold"] = 0.3
    
    @classmethod
    def get_workspace_root(cls) -> Path:
        """Get the workspace root directory"""
        return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    @classmethod
    def get_uploads_dir(cls) -> Path:
        """Get the uploads directory path"""
        return cls.get_workspace_root() / "uploads"
    
    @classmethod
    def get_templates_dir(cls) -> Path:
        """Get the templates directory path"""
        return cls.get_workspace_root() / "templates" 