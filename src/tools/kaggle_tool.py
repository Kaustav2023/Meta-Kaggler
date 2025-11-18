"""
Kaggle dataset ingestion tool - FIXED VERSION
Keeps all columns for Gemini to analyze intelligently
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from config import Config
import subprocess
import os

class KaggleTool:
    """Download and load Kaggle datasets"""
    
    def __init__(self, mock_mode: bool = Config.MOCK_MODE):
        self.mock_mode = mock_mode
    
    def download_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Download dataset (mock or real)"""
        
        if self.mock_mode:
            print(f"📦 MOCK MODE: Generating synthetic data for '{dataset_id}'")
            return self._generate_mock_data(dataset_id)
        else:
            print(f"🌐 REAL MODE: Downloading dataset '{dataset_id}' from Kaggle...")
            return self._download_real_data(dataset_id)
    
    def _generate_mock_data(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic dataset for demo"""
        np.random.seed(Config.RANDOM_STATE)
        
        n_samples = 1000
        n_features = 10
        
        # Create synthetic classification dataset
        df = pd.DataFrame({
            **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
            'target': np.random.randint(0, 2, n_samples)
        })
        
        context = {
            'dataset_id': dataset_id,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'target_column': 'target',
            'task_type': 'classification',
            'sample': df.head(3).to_dict()
        }
        
        print(f"✅ Mock data generated: {df.shape}")
        return df, context
    
    def _download_real_data(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Download actual Kaggle dataset - KEEP ALL COLUMNS for Gemini analysis"""
        
        download_path = Config.ARTIFACTS_DIR / "downloads" / dataset_id.replace('/', '_')
        download_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Check if Kaggle credentials exist
            kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
            if not kaggle_config.exists():
                raise Exception(
                    f"❌ Kaggle API credentials not found!\n"
                    f"Expected at: {kaggle_config}\n"
                    f"Download from: https://www.kaggle.com/settings/account"
                )
            
            print(f"📥 Downloading dataset: {dataset_id}")
            
            # Download using kaggle CLI
            cmd = f'kaggle datasets download -d {dataset_id} -p "{download_path}" --unzip'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_detail = result.stderr if result.stderr else result.stdout
                raise Exception(f"Kaggle download error: {error_detail}")
            
            print(f"✅ Download complete")
            
            # Find CSV file
            csv_files = list(download_path.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {download_path}")
            
            # Use largest CSV file
            csv_file = max(csv_files, key=lambda f: f.stat().st_size)
            print(f"📄 Loading: {csv_file.name}")
            
            # Load CSV - KEEP ALL COLUMNS (numeric AND categorical)
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Minimal cleaning only
            # Drop completely empty rows
            df = df.dropna(how='all', axis=0)
            
            # Drop completely empty columns
            df = df.dropna(how='all', axis=1)
            
            # Remove perfect duplicates
            df = df.drop_duplicates()
            
            if len(df) == 0:
                raise ValueError("Dataset is empty after basic cleaning")
            
            print(f"✅ After basic cleanup: {len(df)} rows, {len(df.columns)} columns")
            
            # Return RAW data with ALL columns
            # Let PreprocessingAgent (powered by Gemini) handle intelligent decisions
            context = {
                'dataset_id': dataset_id,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'sample': df.head(3).to_dict(),
                'original_file': str(csv_file)
            }
            
            return df, context
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            print("\n⚠️  FALLING BACK TO MOCK MODE\n")
            return self._generate_mock_data(dataset_id)
