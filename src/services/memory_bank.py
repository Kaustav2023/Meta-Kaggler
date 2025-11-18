"""
Persistent memory for best models across runs
"""
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from config import Config

class MemoryBank:
    """Store and retrieve best models per dataset"""
    
    def __init__(self, path: Path = Config.MEMORY_BANK_PATH):
        self.path = path
        self.memory = self._load()
    
    def _load(self) -> Dict:
        """Load memory from disk"""
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save(self):
        """Persist memory to disk"""
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def store_best_model(self, dataset_id: str, model_info: Dict):
        """Store best model for a dataset"""
        if dataset_id not in self.memory:
            self.memory[dataset_id] = []
        
        model_info['timestamp'] = datetime.now().isoformat()
        self.memory[dataset_id].append(model_info)
        
        # Keep only top 5 models per dataset
        self.memory[dataset_id] = sorted(
            self.memory[dataset_id],
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:5]
        
        self._save()
    
    def get_best_model(self, dataset_id: str) -> Optional[Dict]:
        """Retrieve best model for a dataset"""
        if dataset_id in self.memory and self.memory[dataset_id]:
            return self.memory[dataset_id][0]
        return None
    
    def get_all_models(self, dataset_id: str) -> List[Dict]:
        """Get all stored models for a dataset"""
        return self.memory.get(dataset_id, [])
