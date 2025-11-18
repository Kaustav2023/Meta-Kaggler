"""
DataPreprocessingAgent - Intelligent data preprocessing powered by Gemini
Analyzes raw data and provides smart preprocessing recommendations
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple
from config import Config

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class DataPreprocessingAgent:
    """Intelligently preprocesses raw data using Gemini LLM"""
    
    def __init__(self, logger):
        self.logger = logger
        self.use_gemini = GEMINI_AVAILABLE and Config.GEMINI_API_KEY
        
        if self.use_gemini:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.logger.info("preprocessing_agent_initialized", mode="Gemini-powered")
        else:
            self.logger.info("preprocessing_agent_initialized", mode="heuristic")
    
    def preprocess(self, df: pd.DataFrame, data_context: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Intelligently preprocess data using Gemini analysis
        
        Args:
            df: Raw dataframe (all columns intact)
            data_context: Dataset metadata
            
        Returns:
            Cleaned dataframe and preprocessing instructions
        """
        self.logger.info("preprocessing_started", shape=df.shape)
        
        if self.use_gemini:
            return self._gemini_preprocess(df, data_context)
        else:
            return self._heuristic_preprocess(df, data_context)
    
    def _gemini_preprocess(self, df: pd.DataFrame, data_context: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Use Gemini to analyze and preprocess data intelligently"""
        
        # Prepare comprehensive data summary for Gemini
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
            'unique_counts': {col: df[col].nunique() for col in df.columns},
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'sample_data': df.head(3).to_dict(orient='records')
        }
        
        prompt = f"""You are an expert data scientist analyzing a raw dataset for machine learning preprocessing.

IMPORTANT: You must work with ALL columns provided. Do not skip analysis of any column.

Dataset Summary:
- Shape: {summary['shape']}
- All Columns: {summary['columns']}
- Data Types: {summary['dtypes']}
- Missing Values: {summary['missing_values']}
- Unique Counts: {summary['unique_counts']}
- Numeric Columns: {summary['numeric_columns']}
- Categorical Columns: {summary['categorical_columns']}

Sample Data (first 3 rows):
{json.dumps(summary['sample_data'], indent=2)}

TASK: Provide intelligent preprocessing recommendations for ML model training.

Return ONLY a valid JSON object (no markdown, no explanation) with this EXACT structure:
{{
  "target_column": "name of best target column for prediction",
  "feature_columns": ["list", "of", "all", "feature", "column", "names"],
  "drop_columns": ["columns", "to", "completely", "remove"],
  "convert_to_numeric": ["categorical", "columns", "to", "encode"],
  "encode_categorical": ["more", "categorical", "columns"],
  "convert_datetime": ["datetime", "columns"],
  "handle_missing": {{
    "strategy": "drop_rows OR fill_mean OR fill_zero",
    "affected_columns": ["col1", "col2"]
  }},
  "remove_duplicates": true,
  "task_type": "classification OR regression",
  "reasoning": "Brief explanation of your preprocessing strategy"
}}

CRITICAL RULES:
1. ALWAYS include categorical columns in feature_columns list - they will be encoded
2. The target column MUST exist in the dataset
3. For Iris: target_column should be 'Species' (will be encoded to numeric)
4. For COVID: choose 'new_cases' or 'new_deaths' as target
5. Never recommend dropping the target column
6. Include Id/index columns in drop_columns list
7. Return ONLY valid JSON, nothing else
"""
        
        try:
            print("🧠 Gemini analyzing dataset structure...")
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown if present
            if "```" in text:
                first_start = text.find("```")
                first_end = text.find("```", first_start + 3)
                if first_end != -1:
                    # Extract inner content between the first pair of fences
                    inner = text[first_start + 3:first_end]
                    # If language tag like "json\n" is present at the start, remove it
                    # e.g. "json\n{...}" -> "{...}"
                    inner = inner.lstrip()
                    # remove a leading language token if present (e.g., "json\n")
                    if "\n" in inner:
                        possible_lang, rest = inner.split("\n", 1)
                        # treat a short token (<= 20 chars, letters) as a language tag
                        if len(possible_lang) <= 20 and possible_lang.isalpha():
                            inner = rest
                    text = inner.strip()
                else:
                    # Unmatched fences — just remove backticks and continue
                    text = text.replace("```", "").strip()
            
            # Parse JSON
            instructions = json.loads(text)
            
            self.logger.info("gemini_analysis_complete", 
                           task_type=instructions.get('task_type'),
                           target=instructions.get('target_column'))
            
            print(f"✅ Gemini recommendations:")
            print(f"   Target: {instructions.get('target_column')}")
            print(f"   Features: {len(instructions.get('feature_columns', []))} columns")
            print(f"   Task: {instructions.get('task_type')}")
            
            # Apply preprocessing
            df_clean = self._apply_preprocessing(df, instructions)
            
            return df_clean, instructions
            
        except Exception as e:
            self.logger.warning("gemini_preprocessing_failed", error=str(e))
            self.logger.info("falling_back_to_heuristic")
            print(f"⚠️  Gemini preprocessing failed: {str(e)}")
            print(f"   Falling back to heuristic approach...")
            return self._heuristic_preprocess(df, data_context)
    
    def _heuristic_preprocess(self, df: pd.DataFrame, data_context: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Fallback heuristic preprocessing"""
        
        df_clean = df.copy()
        
        # Drop rows with all NaN
        df_clean = df_clean.dropna(how='all')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Identify numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical columns to numeric
        for col in categorical_cols:
            df_clean[col] = pd.Categorical(df_clean[col]).codes
        
        all_cols = numeric_cols + categorical_cols
        
        if len(all_cols) < 2:
            raise ValueError("Not enough columns for ML")
        
        # Target = last column
        target_col = all_cols[-1]
        feature_cols = all_cols[:-1]
        
        instructions = {
            'target_column': target_col,
            'feature_columns': feature_cols,
            'drop_columns': [col for col in df_clean.columns if col not in all_cols],
            'convert_to_numeric': categorical_cols,
            'encode_categorical': [],
            'convert_datetime': [],
            'handle_missing': {'strategy': 'drop_rows', 'affected_columns': []},
            'remove_duplicates': True,
            'task_type': 'classification' if df_clean[target_col].nunique() < 20 else 'regression',
            'reasoning': 'Heuristic: dropped non-numeric columns, encoded categoricals'
        }
        
        return df_clean[feature_cols + [target_col]], instructions
    
    def _apply_preprocessing(self, df: pd.DataFrame, instructions: Dict) -> pd.DataFrame:
        """Apply Gemini's preprocessing instructions"""
        
        df_clean = df.copy()
        
        print("\n📋 Applying preprocessing steps...")
        
        # Get target column
        target_col = instructions.get('target_column')
        if not target_col or target_col not in df_clean.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {df_clean.columns.tolist()}")
        
        # 1. Encode categorical TARGET column first (if needed)
        if target_col in df_clean.columns and df_clean[target_col].dtype == 'object':
            print(f"   🏷️  Encoding target column '{target_col}'")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_clean[target_col] = le.fit_transform(df_clean[target_col].astype(str))
        
        # 2. Drop columns (except target)
        drop_cols = [col for col in instructions.get('drop_columns', []) 
                     if col in df_clean.columns and col != target_col]
        if drop_cols:
            print(f"   🗑️  Dropping columns: {drop_cols}")
            df_clean = df_clean.drop(columns=drop_cols)
        
        # 3. Convert to numeric
        convert_cols = instructions.get('convert_to_numeric', [])
        for col in convert_cols:
            if col in df_clean.columns and col != target_col:
                print(f"   🔢 Converting {col} to numeric")
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 4. Handle datetime columns
        date_cols = instructions.get('convert_datetime', [])
        for col in date_cols:
            if col in df_clean.columns and col != target_col:
                print(f"   📅 Parsing {col} as datetime")
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                df_clean = df_clean.drop(columns=[col])
        
        # 5. Encode categorical feature columns
        cat_cols = instructions.get('encode_categorical', [])
        for col in cat_cols:
            if col in df_clean.columns and col != target_col:
                print(f"   🏷️  Encoding {col}")
                df_clean[col] = pd.Categorical(df_clean[col]).codes
        
        # 6. Handle missing values
        missing_strategy = instructions.get('handle_missing', {}).get('strategy', 'drop_rows')
        if missing_strategy == 'drop_rows':
            print(f"   ❌ Dropping rows with missing values")
            df_clean = df_clean.dropna()
        elif missing_strategy == 'fill_mean':
            print(f"   📊 Filling missing values with mean")
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif missing_strategy == 'fill_zero':
            print(f"   0️⃣  Filling missing values with zero")
            df_clean = df_clean.fillna(0)
        
        # 7. Remove duplicates
        if instructions.get('remove_duplicates', False):
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = before - len(df_clean)
            if removed > 0:
                print(f"   🔄 Removed {removed} duplicate rows")
        
        # 8. Select feature columns + target (only numeric ones that exist)
        feature_cols = instructions.get('feature_columns', [])
        
        # Keep only columns that exist in dataframe
        existing_features = [col for col in feature_cols 
                            if col in df_clean.columns and col != target_col]
        
        if not existing_features:
            # Fallback: use all numeric columns except target
            numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
            existing_features = [col for col in numeric_cols if col != target_col]
        
        if not existing_features:
            raise ValueError("No feature columns available for training")
        
        # Final dataframe: features + target
        final_cols = existing_features + [target_col]
        df_clean = df_clean[final_cols]
        
        print(f"\n✅ Preprocessing complete: {df_clean.shape}")
        
        return df_clean
