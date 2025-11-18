"""
DataAgent - Dataset ingestion and profiling
Now uses DataPreprocessingAgent for intelligent cleaning
"""
import pandas as pd
from typing import Dict, Tuple
from src.tools.kaggle_tool import KaggleTool
from src.agents.preprocessing_agent import DataPreprocessingAgent

class DataAgent:
    """Handles dataset download and profiling"""
    
    def __init__(self, logger, mock_mode: bool = False):
        self.logger = logger
        self.mock_mode = mock_mode
        self.kaggle_tool = KaggleTool(mock_mode=mock_mode)
        self.preprocessing_agent = DataPreprocessingAgent(logger)  # ← NEW!
    
    def load_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load, profile, and intelligently preprocess dataset
        """
        self.logger.info("data_agent_started", dataset_id=dataset_id, mock_mode=self.mock_mode)
        
        # Step 1: Download dataset
        df, context = self.kaggle_tool.download_dataset(dataset_id)
        
        # Step 2: Intelligent preprocessing (Gemini-powered)
        print(f"\n🔍 Preprocessing dataset with AI analysis...")
        df_clean, preprocessing_instructions = self.preprocessing_agent.preprocess(df, context)
        
        # Step 3: Validate preprocessed data
        if df_clean is None or len(df_clean) == 0:
            raise ValueError(f"Failed to preprocess dataset: {dataset_id}")
        
        # Step 4: Update context with preprocessing results
        context['shape_original'] = context['shape']
        context['shape'] = df_clean.shape
        context['target_column'] = preprocessing_instructions.get('target_column')
        context['task_type'] = preprocessing_instructions.get('task_type')
        context['preprocessing'] = preprocessing_instructions
        
        # Print summary
        print(f"\n📊 Final Dataset Summary:")
        print(f"   Original Shape: {context['shape_original']}")
        print(f"   Final Shape: {context['shape']}")
        print(f"   Target: {context['target_column']}")
        print(f"   Task: {context['task_type']}")
        print(f"   Features: {context['shape'][1] - 1}\n")
        
        self.logger.info("data_loaded_and_preprocessed", 
                        shape=context['shape'],
                        target=context.get('target_column'))
        
        return df_clean, context
