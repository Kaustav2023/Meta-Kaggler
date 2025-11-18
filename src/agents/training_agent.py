"""
TrainingAgent - Parallel model training execution
Trains multiple ML models concurrently using ThreadPoolExecutor
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import joblib
from pathlib import Path
from config import Config

class TrainingAgent:
    """Trains multiple models in parallel"""
    
    def __init__(self, logger):
        self.logger = logger
        
        # Model registry for instantiation
        self.model_registry = {
            'RandomForestClassifier': RandomForestClassifier,
            'XGBClassifier': XGBClassifier,
            'LGBMClassifier': LGBMClassifier,
            'LogisticRegression': LogisticRegression
        }
    
    def train_all_models(self, df: pd.DataFrame, proposals: List[Dict], data_context: Dict) -> List[Dict]:
        """
        Train all proposed models in parallel
        
        Args:
            df: Dataset DataFrame
            proposals: List of model proposals from ModelAgent
            data_context: Dataset metadata
            
        Returns:
            List of training results sorted by score
        """
        self.logger.info("training_started", num_models=len(proposals))
        
        # Prepare train/validation split
        X_train, y_train, X_val, y_val = self._prepare_data(df, data_context)
        
        # Train models in parallel using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=Config.MAX_PARALLEL_TRAINING) as executor:
            # Submit all training jobs
            futures = {
                executor.submit(self._train_single_model, proposal, X_train, y_train, X_val, y_val): proposal
                for proposal in proposals
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                proposal = futures[future]
                try:
                    result = future.result(timeout=Config.MODEL_TIMEOUT_SECONDS)
                    results.append(result)
                    self.logger.info("model_trained",
                                   model_id=result['model_id'],
                                   score=result['score'],
                                   time=result['training_time'])
                except Exception as e:
                    self.logger.error("training_failed",
                                    model_id=proposal['model_id'],
                                    error=str(e))
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _prepare_data(self, df: pd.DataFrame, data_context: Dict):
        """Split data into train/validation sets"""
        target_col = data_context.get('target_column', 'target')
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Sample if dataset too large
        if len(X) > Config.MAX_TRAINING_SAMPLES:
            X = X.sample(Config.MAX_TRAINING_SAMPLES, random_state=Config.RANDOM_STATE)
            y = y.loc[X.index]
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=Config.TRAIN_TEST_SPLIT,
            random_state=Config.RANDOM_STATE
        )
        
        return X_train, y_train, X_val, y_val
    
    def _train_single_model(self, proposal: Dict, X_train, y_train, X_val, y_val) -> Dict:
        """
        Train a single model
        
        Args:
            proposal: Model configuration
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training result with score and metadata
        """
        start_time = time.time()
        
        # Instantiate model from registry
        model_class = self.model_registry[proposal['model_type']]
        model = model_class(**proposal['params'])
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        # Save trained model artifact
        model_path = Config.MODELS_DIR / f"{proposal['model_id']}.joblib"
        joblib.dump(model, model_path)
        
        training_time = time.time() - start_time
        
        return {
            'model_id': proposal['model_id'],
            'model_type': proposal['model_type'],
            'params': proposal['params'],
            'score': round(score, 4),
            'training_time': round(training_time, 2),
            'artifact_path': str(model_path)
        }
