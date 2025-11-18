"""
CoordinatorAgent - Main orchestrator with Agent Evaluation
Coordinates all agents to execute the full ML pipeline and evaluates performance
"""
from typing import Dict
from src.agents.data_agent import DataAgent
from src.agents.model_agent import ModelAgent
from src.agents.training_agent import TrainingAgent
from src.services.memory_bank import MemoryBank
from src.services.session_service import InMemorySessionService, SessionStatus
from src.services.agent_evaluator import AgentEvaluator
from config import Config

class CoordinatorAgent:
    """Main orchestrator for the ML pipeline with evaluation"""
    
    def __init__(self, logger, use_llm: bool = False, mock_mode: bool = False):
        self.logger = logger
        
        # Initialize all agents - PASS mock_mode explicitly!
        self.data_agent = DataAgent(logger, mock_mode=mock_mode)
        self.model_agent = ModelAgent(logger, use_llm=use_llm)
        self.training_agent = TrainingAgent(logger)
        
        # Initialize services
        self.memory_bank = MemoryBank()
        self.session_service = InMemorySessionService()
        self.evaluator = AgentEvaluator(logger)
    
    def run_experiment(self, dataset_id: str) -> Dict:
        """
        Execute full ML pipeline with agent evaluation
        
        Pipeline steps:
        1. Load dataset (DataAgent)
        2. Preprocess data (PreprocessingAgent via DataAgent)
        3. Propose models (ModelAgent with optional Gemini)
        4. Train models in parallel (TrainingAgent)
        5. Select best model
        6. Store in MemoryBank
        7. Evaluate agent performance
        
        Args:
            dataset_id: Kaggle dataset identifier
            
        Returns:
            Experiment results with best model, all results, and evaluation
        """
        # Create session to track experiment
        run_id = self.session_service.create_session(dataset_id, {
            'mock_mode': Config.MOCK_MODE,
            'use_llm': Config.USE_LLM
        })
        self.logger.info("experiment_started", run_id=run_id, dataset_id=dataset_id)
        
        try:
            # Step 1: Load and profile dataset
            self.session_service.update_status(run_id, SessionStatus.STARTED)
            df, data_context = self.data_agent.load_dataset(dataset_id)
            self.session_service.update_status(run_id, SessionStatus.DATA_LOADED, 
                                              {'shape': data_context['shape'],
                                               'shape_original': data_context.get('shape_original')})
            
            # Validate dataset has features
            if data_context['shape'][1] < 2:  # Need at least 1 feature + 1 target
                raise ValueError(f"Dataset has insufficient columns: {data_context['shape']}")
            
            # Step 2: Generate model proposals (LLM or heuristic)
            proposals = self.model_agent.propose_models(data_context)
            self.session_service.update_status(run_id, SessionStatus.MODELS_PROPOSED, 
                                              {'count': len(proposals)})
            
            # Step 3: Train all models in parallel
            self.session_service.update_status(run_id, SessionStatus.TRAINING)
            results = self.training_agent.train_all_models(df, proposals, data_context)
            
            # Check if any models trained successfully
            if not results or len(results) == 0:
                print("\n⚠️  WARNING: No models trained successfully!")
                print("   Possible causes:")
                print("   - Dataset has no valid features")
                print("   - All target values are NaN")
                print("   - Insufficient data for training")
                print("   - Model compatibility issues")
                raise ValueError("All models failed to train. Check data quality and format.")
            
            # Step 4: Select best model (already sorted by score)
            best_model = results[0]
            
            # Add shape info to best_model for evaluation
            best_model['shape'] = data_context['shape']
            
            # Step 5: Store best model in MemoryBank
            self.memory_bank.store_best_model(dataset_id, best_model)
            
            # Complete session
            self.session_service.update_status(run_id, SessionStatus.COMPLETED)
            self.session_service.store_results(run_id, {
                'best_model': best_model,
                'all_results': results
            })
            
            # Build result dictionary
            result = {
                'run_id': run_id,
                'dataset_id': dataset_id,
                'best_model': best_model,
                'all_results': results,
                'session': self.session_service.get_session(run_id)
            }
            
            # Step 6: EVALUATE AGENTS
            evaluation = self.evaluator.evaluate_experiment(result)
            result['evaluation'] = evaluation
            
            self.logger.info("experiment_completed",
                           run_id=run_id,
                           best_model=best_model['model_id'],
                           best_score=best_model['score'],
                           overall_agent_score=evaluation['overall_score'])
            
            return result
            
        except Exception as e:
            self.session_service.update_status(run_id, SessionStatus.FAILED, 
                                              {'error': str(e)})
            self.logger.error("experiment_failed", run_id=run_id, error=str(e))
            
            # Return error result instead of raising
            return {
                'run_id': run_id,
                'dataset_id': dataset_id,
                'status': 'failed',
                'error': str(e),
                'best_model': None,
                'all_results': [],
                'session': self.session_service.get_session(run_id),
                'evaluation': None
            }
