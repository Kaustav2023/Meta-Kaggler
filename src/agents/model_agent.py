"""
ModelAgent - ML model proposal generation
Uses Gemini LLM for intelligent proposals with heuristic fallback
"""
from typing import List, Dict
from config import Config
import json

# Try to import Gemini, fallback if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ModelAgent:
    """Proposes multiple ML models to try"""
    
    def __init__(self, logger, use_llm: bool = True):
        self.logger = logger
        self.use_llm = use_llm and GEMINI_AVAILABLE and Config.GEMINI_API_KEY
        
        if self.use_llm:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.logger.info("model_agent_initialized", mode="LLM-powered (Gemini)")
        else:
            self.logger.info("model_agent_initialized", mode="heuristic")
    
    def propose_models(self, data_context: Dict) -> List[Dict]:
        """
        Generate model proposals using LLM or heuristics
        
        Args:
            data_context: Dataset characteristics and metadata
            
        Returns:
            List of model proposal dicts with id, type, and params
        """
        self.logger.info("model_agent_started", 
                        num_proposals=Config.NUM_MODEL_PROPOSALS,
                        use_llm=self.use_llm)
        
        if self.use_llm:
            return self._llm_proposals(data_context)
        else:
            return self._heuristic_proposals(data_context)
    
    def _llm_proposals(self, data_context: Dict) -> List[Dict]:
        """Use Gemini LLM to generate intelligent model proposals"""
        
        prompt = f"""You are an expert ML engineer analyzing a Kaggle dataset.

Dataset Context:
- Shape: {data_context['shape']}
- Task: {data_context.get('task_type', 'classification')}
- Features: {len(data_context.get('numeric_columns', []))} numeric, {len(data_context.get('categorical_columns', []))} categorical
- Target: {data_context.get('target_column')}

Propose {Config.NUM_MODEL_PROPOSALS} diverse machine learning models for this dataset.

Return ONLY a JSON array with this exact structure:
[
  {{
    "model_id": "unique_id",
    "model_type": "RandomForestClassifier|XGBClassifier|LGBMClassifier|LogisticRegression",
    "params": {{"n_estimators": 100, "max_depth": 10, "random_state": 42}}
  }}
]

IMPORTANT:
- Only use: RandomForestClassifier, XGBClassifier, LGBMClassifier, LogisticRegression
- model_id should be descriptive (e.g., "rf_shallow", "xgb_tuned")
- Always include random_state: 42 in params
- Vary hyperparameters across proposals
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown code blocks if present
            if "```json" in text:
                text = text.split("``````")[0].strip()
            elif "```" in text:
                text = text.split("``````")[0].strip()
            
            proposals = json.loads(text)
            
            self.logger.info("llm_proposals_generated", count=len(proposals))
            
            # Validate and return
            return [
                {
                    'model_id': p['model_id'],
                    'model_type': p['model_type'],
                    'params': p['params']
                }
                for p in proposals[:Config.NUM_MODEL_PROPOSALS]
            ]
            
        except Exception as e:
            self.logger.warning("llm_proposal_failed", error=str(e))
            self.logger.info("falling_back_to_heuristic")
            return self._heuristic_proposals(data_context)
    
    def _heuristic_proposals(self, data_context: Dict) -> List[Dict]:
        """Fallback heuristic proposals (no LLM needed)"""
        
        proposals = [
            {
                'model_id': 'rf_default',
                'model_type': 'RandomForestClassifier',
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': Config.RANDOM_STATE}
            },
            {
                'model_id': 'rf_deep',
                'model_type': 'RandomForestClassifier',
                'params': {'n_estimators': 200, 'max_depth': 20, 'random_state': Config.RANDOM_STATE}
            },
            {
                'model_id': 'xgb_default',
                'model_type': 'XGBClassifier',
                'params': {'n_estimators': 100, 'max_depth': 6, 'random_state': Config.RANDOM_STATE, 'eval_metric': 'logloss'}
            },
            {
                'model_id': 'xgb_tuned',
                'model_type': 'XGBClassifier',
                'params': {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': Config.RANDOM_STATE, 'eval_metric': 'logloss'}
            },
            {
                'model_id': 'lgbm_default',
                'model_type': 'LGBMClassifier',
                'params': {'n_estimators': 100, 'random_state': Config.RANDOM_STATE, 'verbose': -1}
            },
            {
                'model_id': 'logreg',
                'model_type': 'LogisticRegression',
                'params': {'max_iter': 1000, 'random_state': Config.RANDOM_STATE}
            }
        ]
        
        self.logger.info("heuristic_proposals_generated", count=len(proposals))
        return proposals[:Config.NUM_MODEL_PROPOSALS]
