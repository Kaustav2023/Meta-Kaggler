"""
In-memory session tracking
"""
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import uuid

class SessionStatus(Enum):
    """Session lifecycle states"""
    INITIALIZED = "initialized"
    STARTED = "started"
    DATA_LOADED = "data_loaded"
    MODELS_PROPOSED = "models_proposed"
    TRAINING = "training"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"

class InMemorySessionService:
    """Track experiment session state"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, dataset_id: str, config: Dict) -> str:
        """Create new session"""
        run_id = str(uuid.uuid4())[:8]
        
        self.sessions[run_id] = {
            'run_id': run_id,
            'dataset_id': dataset_id,
            'config': config,
            'status': SessionStatus.INITIALIZED.value,
            'created_at': datetime.now().isoformat(),
            'steps': [],
            'results': {},
            'metadata': {}
        }
        
        return run_id
    
    def update_status(self, run_id: str, status: SessionStatus, data: Optional[Dict] = None):
        """Update session status"""
        if run_id in self.sessions:
            self.sessions[run_id]['status'] = status.value
            self.sessions[run_id]['updated_at'] = datetime.now().isoformat()
            
            if data:
                self.sessions[run_id]['steps'].append({
                    'status': status.value,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                })
    
    def store_results(self, run_id: str, results: Dict):
        """Store final results"""
        if run_id in self.sessions:
            self.sessions[run_id]['results'] = results
    
    def get_session(self, run_id: str) -> Optional[Dict]:
        """Retrieve session"""
        return self.sessions.get(run_id)
    
    def pause_session(self, run_id: str):
        """Pause session"""
        self.update_status(run_id, SessionStatus.PAUSED)
    
    def resume_session(self, run_id: str):
        """Resume session"""
        self.update_status(run_id, SessionStatus.RESUMED)
