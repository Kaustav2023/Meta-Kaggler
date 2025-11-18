"""
Main entry point for Autonomous Kaggle Competition Companion
With Agent Evaluation System
"""
import argparse
from config import Config
from src.agents.coordinator_agent import CoordinatorAgent
from src.utils.logger import setup_logger
import json

def main():
    """Execute ML pipeline with agent evaluation"""
    parser = argparse.ArgumentParser(
        description="Autonomous Kaggle Competition Companion - Multi-Agent ML Automation"
    )
    parser.add_argument('--dataset', type=str, default='demo-dataset',
                       help='Kaggle dataset ID (e.g., "yasserh/titanic-dataset")')
    parser.add_argument('--mock', action='store_true',
                       help='Run in mock mode (no Kaggle API required)')
    parser.add_argument('--use-llm', action='store_true',
                       help='Use Gemini LLM for intelligent model proposals')
    
    args = parser.parse_args()
    
    # Configure system
    Config.MOCK_MODE = args.mock
    Config.USE_LLM = args.use_llm
    Config.create_directories()
    
    # Setup logging
    logger = setup_logger(f"run_{args.dataset.replace('/', '_')}")
    
    # Initialize coordinator - PASS mock_mode explicitly!
    coordinator = CoordinatorAgent(logger, use_llm=args.use_llm, mock_mode=args.mock)
    
    # Print header
    print("\n" + "="*80)
    print("🤖 Autonomous Kaggle Competition Companion")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {'MOCK' if Config.MOCK_MODE else 'REAL'}")
    print(f"Model Proposals: {'Gemini-powered' if Config.USE_LLM else 'Heuristic'}")
    print("="*80 + "\n")
    
    try:
        # Execute experiment
        result = coordinator.run_experiment(args.dataset)
        
        # Check if experiment failed
        if result.get('status') == 'failed':
            print("\n" + "="*80)
            print("❌ EXPERIMENT FAILED")
            print("="*80)
            print(f"\nError: {result.get('error', 'Unknown error')}\n")
            print("="*80 + "\n")
            return
        
        # Check if results are empty (all models failed)
        if not result.get('all_results') or len(result.get('all_results', [])) == 0:
            print("\n" + "="*80)
            print("❌ NO MODELS TRAINED SUCCESSFULLY")
            print("="*80)
            print("\nAll models failed during training.")
            print("Possible causes:")
            print("  - Dataset has insufficient features")
            print("  - Data quality issues (missing values, wrong types)")
            print("  - Model compatibility with data type\n")
            print("="*80 + "\n")
            return
        
        # Display training results
        print("\n" + "="*80)
        print("📊 TRAINING RESULTS")
        print("="*80)
        
        best_model = result.get('best_model')
        if best_model:
            print(f"\n🏆 Best Model: {best_model['model_id']}")
            print(f"   Type: {best_model['model_type']}")
            print(f"   Score: {best_model['score']}")
            print(f"   Training Time: {best_model['training_time']}s")
            print(f"   Artifact: {best_model['artifact_path']}")
        
        print("\n📈 All Models:")
        print("-" * 80)
        for i, model in enumerate(result['all_results'], 1):
            print(f"{i}. {model['model_id']:20} | Type: {model['model_type']:25} | Score: {model['score']:.4f} | Time: {model['training_time']}s")
        
        print("\n" + "="*80)
        print(f"✅ Training Complete! Run ID: {result['run_id']}")
        print("="*80 + "\n")
        
        # Display agent evaluation
        if result.get('evaluation'):
            evaluation = result['evaluation']
            
            print("\n" + "="*80)
            print("🤖 AGENT EVALUATION")
            print("="*80)
            
            # Display each agent's evaluation
            for agent_name, agent_eval in evaluation.get('agents_evaluated', {}).items():
                score = agent_eval.get('score', 0)
                status = agent_eval.get('status', 'UNKNOWN')
                
                # Emoji based on status
                emoji = "✅" if status == "PASS" else "⚠️"
                
                print(f"\n{emoji} {agent_name}:")
                print(f"   Score: {score:.2f} [{status}]")
                
                # Display metrics
                metrics = agent_eval.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    # Format metric value
                    if isinstance(metric_value, float):
                        formatted_value = f"{metric_value:.2f}"
                    elif isinstance(metric_value, list):
                        formatted_value = f"{len(metric_value)} items"
                    else:
                        formatted_value = str(metric_value)
                    
                    print(f"   • {metric_name}: {formatted_value}")
            
            # Overall score
            overall_score = evaluation['overall_score']
            overall_emoji = "🏆" if overall_score >= 0.8 else "✅" if overall_score >= 0.6 else "⚠️"
            
            print(f"\n{overall_emoji} Overall Agent Score: {overall_score:.2f}/1.0")
            
            # Recommendations
            recommendations = evaluation.get('recommendations', [])
            if recommendations:
                print("\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"   {rec}")
            
            print("\n" + "="*80 + "\n")
        
        # Save results to JSON
        results_file = Config.ARTIFACTS_DIR / f"results_{result['run_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"📄 Results saved to: {results_file}")
        
        # Display dataset info
        session = result.get('session', {})
        if session:
            print(f"\n📊 Dataset Info:")
            print(f"   Dataset ID: {session.get('dataset_id')}")
            print(f"   Status: {session.get('status')}")
            
            # Show original vs final shape
            metadata = session.get('metadata', {})
            if metadata.get('shape_original'):
                print(f"   Original Shape: {metadata.get('shape_original')}")
                print(f"   Final Shape: {metadata.get('shape')}")
            print()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return
    
    except Exception as e:
        print("\n" + "="*80)
        print("❌ UNEXPECTED ERROR")
        print("="*80)
        print(f"\nError: {str(e)}\n")
        print("="*80 + "\n")
        raise

if __name__ == "__main__":
    main()
