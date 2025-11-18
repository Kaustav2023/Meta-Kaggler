"""
AgentEvaluator - Evaluates agent performance and decision quality
Non-invasive evaluation that doesn't affect core pipeline
"""
import json
from typing import Dict, List
from pathlib import Path
from config import Config
from datetime import datetime

class AgentEvaluator:
    """Evaluates agent performance metrics"""
    
    def __init__(self, logger):
        self.logger = logger
        self.evaluation_dir = Config.ARTIFACTS_DIR / "evaluations"
        self.evaluation_dir.mkdir(exist_ok=True)
    
    def evaluate_experiment(self, result: Dict) -> Dict:
        """
        Evaluate the entire experiment
        
        Args:
            result: Experiment result from CoordinatorAgent
            
        Returns:
            Evaluation metrics
        """
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'run_id': result.get('run_id'),
            'dataset_id': result.get('dataset_id'),
            'agents_evaluated': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # 1. Evaluate DataAgent
        evaluation['agents_evaluated']['DataAgent'] = self._evaluate_data_agent(result)
        
        # 2. Evaluate PreprocessingAgent
        evaluation['agents_evaluated']['PreprocessingAgent'] = self._evaluate_preprocessing_agent(result)
        
        # 3. Evaluate ModelAgent
        evaluation['agents_evaluated']['ModelAgent'] = self._evaluate_model_agent(result)
        
        # 4. Evaluate TrainingAgent
        evaluation['agents_evaluated']['TrainingAgent'] = self._evaluate_training_agent(result)
        
        # 5. Calculate overall score
        agent_scores = [v.get('score', 0) for v in evaluation['agents_evaluated'].values()]
        evaluation['overall_score'] = round(sum(agent_scores) / len(agent_scores), 2) if agent_scores else 0.0
        
        # 6. Generate recommendations
        evaluation['recommendations'] = self._generate_recommendations(evaluation)
        
        # Save evaluation
        self._save_evaluation(evaluation)
        
        self.logger.info("experiment_evaluated", 
                        run_id=result.get('run_id'),
                        overall_score=evaluation['overall_score'])
        
        return evaluation
    
    def _evaluate_data_agent(self, result: Dict) -> Dict:
        """Evaluate DataAgent's performance"""
        session = result.get('session', {})
        
        # Check if data was successfully loaded
        success = session.get('status') == 'COMPLETED'
        
        evaluation = {
            'agent': 'DataAgent',
            'metrics': {
                'data_loaded': success,
                'dataset_id': result.get('dataset_id'),
                'rows': result.get('best_model', {}).get('shape', [0])[0] if result.get('best_model') else 0,
            },
            'score': 1.0 if success else 0.5,
            'status': 'PASS' if success else 'WARN'
        }
        
        return evaluation
    
    def _evaluate_preprocessing_agent(self, result: Dict) -> Dict:
        """Evaluate PreprocessingAgent's intelligence"""
        session = result.get('session', {})
        
        # Check preprocessing quality
        original_shape = session.get('metadata', {}).get('shape_original', (0, 0))
        final_shape = result.get('best_model', {}).get('shape', (0, 0)) if result.get('best_model') else (0, 0)
        
        # Good preprocessing reduces dimensionality intelligently
        dimensionality_score = 1.0
        if original_shape[1] > 0 and final_shape[1] > 0:
            reduction = (original_shape[1] - final_shape[1]) / original_shape[1]
            # Score: too much reduction (>70%) = bad, little reduction (<10%) = good
            if reduction > 0.7:
                dimensionality_score = 0.6
            elif reduction < 0.1:
                dimensionality_score = 0.9
            else:
                dimensionality_score = 0.8
        
        evaluation = {
            'agent': 'PreprocessingAgent',
            'metrics': {
                'original_features': original_shape[1],
                'final_features': final_shape[1],
                'dimensionality_reduction': round((original_shape[1] - final_shape[1]) / max(original_shape[1], 1), 2),
                'rows_retained': final_shape[0] / max(original_shape[0], 1) if original_shape[0] > 0 else 1.0
            },
            'score': round(dimensionality_score, 2),
            'status': 'PASS' if dimensionality_score > 0.6 else 'WARN'
        }
        
        return evaluation
    
    def _evaluate_model_agent(self, result: Dict) -> Dict:
        """Evaluate ModelAgent's proposals"""
        all_results = result.get('all_results', [])
        best_model = result.get('best_model', {})
        
        # Evaluate proposal diversity
        model_types = set()
        for model in all_results:
            model_types.add(model.get('model_type', 'Unknown'))
        
        diversity_score = min(len(model_types) / 4, 1.0)  # 4 model types = perfect diversity
        
        # Evaluate best model performance
        best_score = best_model.get('score', 0)
        performance_score = min(best_score * 2, 1.0)  # 0.5+ = perfect
        
        evaluation = {
            'agent': 'ModelAgent',
            'metrics': {
                'proposals_count': len(all_results),
                'model_types': list(model_types),
                'diversity': len(model_types),
                'best_model_score': best_score
            },
            'score': round((diversity_score + performance_score) / 2, 2),
            'status': 'PASS' if performance_score > 0.5 else 'WARN'
        }
        
        return evaluation
    
    def _evaluate_training_agent(self, result: Dict) -> Dict:
        """Evaluate TrainingAgent's execution"""
        all_results = result.get('all_results', [])
        best_model = result.get('best_model', {})
        
        # Success rate
        success_rate = len(all_results) / 6 if len(all_results) > 0 else 0  # 6 models should train
        
        # Performance consistency (lower variance = more stable)
        if len(all_results) > 1:
            scores = [m.get('score', 0) for m in all_results]
            variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            consistency_score = 1.0 / (1.0 + variance)
        else:
            consistency_score = 0.5
        
        evaluation = {
            'agent': 'TrainingAgent',
            'metrics': {
                'models_trained': len(all_results),
                'success_rate': round(success_rate, 2),
                'best_model': best_model.get('model_id'),
                'best_score': best_model.get('score'),
                'avg_training_time': round(sum(m.get('training_time', 0) for m in all_results) / len(all_results), 2) if all_results else 0
            },
            'score': round((success_rate + consistency_score) / 2, 2),
            'status': 'PASS' if success_rate > 0.5 else 'WARN'
        }
        
        return evaluation
    
    def _generate_recommendations(self, evaluation: Dict) -> List[str]:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        # Analyze each agent's performance
        for agent_name, agent_eval in evaluation['agents_evaluated'].items():
            score = agent_eval.get('score', 0)
            
            if score < 0.6:
                recommendations.append(f"⚠️  {agent_name} scored {score}: Review decision logic")
            elif score > 0.9:
                recommendations.append(f"✅ {agent_name} performing excellently ({score})")
        
        # Overall recommendations
        if evaluation['overall_score'] > 0.8:
            recommendations.append("🏆 Overall agent system performing well - ready for production")
        elif evaluation['overall_score'] < 0.6:
            recommendations.append("🔧 Consider reviewing preprocessing or model selection logic")
        
        return recommendations
    
    def _save_evaluation(self, evaluation: Dict):
        """Save evaluation results"""
        run_id = evaluation.get('run_id', 'unknown')
        eval_file = self.evaluation_dir / f"eval_{run_id}.json"
        
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        self.logger.info("evaluation_saved", path=str(eval_file))
