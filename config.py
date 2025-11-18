"""
Configuration for Kaggle Companion
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    LOGS_DIR = ARTIFACTS_DIR / "logs"
    SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
    MEMORY_BANK_PATH = ARTIFACTS_DIR / "memory_bank.json"
    
    # API Keys (optional)
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # System Settings
    MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
    USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
    NUM_MODEL_PROPOSALS = 6
    MAX_PARALLEL_TRAINING = 4
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Model Configuration
    MODEL_TIMEOUT_SECONDS = 300  # 5 minutes per model
    MAX_TRAINING_SAMPLES = 10000  # For large datasets
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.ARTIFACTS_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.SUBMISSIONS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
