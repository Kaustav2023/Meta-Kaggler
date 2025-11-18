"""
Structured logging configuration
"""
import structlog
import logging
from pathlib import Path
from datetime import datetime
from config import Config

Config.create_directories()

def setup_logger(run_id: str):
    """Setup structured logger for a run"""
    
    # Create log file for this run
    log_file = Config.LOGS_DIR / f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=open(log_file, 'a')),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger()
    logger.info("logger_initialized", run_id=run_id, log_file=str(log_file))
    return logger
