"""
MILP Initial Basis Prediction GNN Model Evaluation Script

This script evaluates the performance of a trained GNN model in predicting the initial basis of LP relaxation for MILP problems, compared to the default baseline of Gurobi.
"""

import os
import sys
import logging
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add the project root directory to the system path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.Initial_basis_prediction.model import InitialBasisGNN
from metrics.Initial_basis_prediction.evaluator import GurobiEvaluator

# Set the log format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="initial_basis_prediction")
def main(cfg: DictConfig) -> None:
    """
    Main function: Evaluate the performance of a trained GNN model in predicting the initial basis of LP relaxation for MILP problems
    
    Args:
        cfg: Hydra configuration object
    """
    # ==================== Initialize ====================
    # Output configuration information
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set the calculation device
    device = torch.device(cfg.model.device if torch.cuda.is_available() and "cuda" in cfg.model.device else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create the results output directory
    results_dir = Path(cfg.evaluation.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # ==================== Path check ====================
    # Determine the model path
    if hasattr(cfg, 'custom_models') and hasattr(cfg.custom_models, 'model_path') and cfg.custom_models.model_path:
        model_path = Path(cfg.custom_models.model_path)
        logger.info(f"Using custom model path: {model_path}")
    else:
        model_path = Path(cfg.training.save_dir) / "basis_prediction_model.pth"
        logger.info(f"Using default model path: {model_path}")
    
    # Check if the model file exists
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return
    
    # Determine the test set path
    test_dir = Path(cfg.datasets.test_dir)
    if not test_dir.exists():
        logger.error(f"Test set path does not exist: {test_dir}")
        return
    
    # Define the result output file path
    output_file = results_dir / "evaluation_results.json"
    
    # ==================== Initialize the evaluator ====================
    # Evaluation configuration
    evaluation_config = {
        'time_limit': cfg.evaluation.time_limit,    # Gurobi solution time limit
        'mip_gap': cfg.evaluation.mip_gap,         # MIP gap setting
        'threads': cfg.evaluation.threads,         # Number of parallel threads
        'results_dir': str(results_dir)            # Results output directory
    }
    
    # Create the evaluator and output the configuration
    dummy_model = InitialBasisGNN()  # Initialize a temporary model, which will be replaced later
    evaluator = GurobiEvaluator(dummy_model, device=device, config=evaluation_config)
    logger.info(f"Gurobi evaluation parameters: time limit={cfg.evaluation.time_limit} seconds, "
               f"MIP gap={cfg.evaluation.mip_gap}, threads={cfg.evaluation.threads}")
    
    # ==================== Load the model ====================
    # Load the pre-trained model
    logger.info(f"Loading model from {model_path}...")
    try:
        # Handle device compatibility
        if not torch.cuda.is_available() and "cuda" in cfg.model.device:
            # If CUDA is not available but the configuration requires GPU, force CPU usage
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            logger.info("CUDA not available, using CPU to load model")
        else:
            checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model structure information
        config = checkpoint.get('config', {})
        model_config = config.get('model', cfg.model)
        
        # Create model instance
        model = InitialBasisGNN(
            var_feat_dim=8,                           # Variable node feature dimension
            constr_feat_dim=8,                        # Constraint node feature dimension
            hidden_dim=getattr(model_config, 'hidden_dim', cfg.model.hidden_dim),  # Hidden layer dimension
            num_layers=getattr(model_config, 'num_layers', cfg.model.num_layers),  # Number of GNN layers
            dropout=getattr(model_config, 'dropout', cfg.model.dropout)            # Dropout ratio
        )
        
        # Load model weights and set evaluation mode
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Replace the model in the evaluator
        evaluator.model = model
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None
    
    # ==================== Execute evaluation ====================
    # Evaluate the model's performance on the test set
    logger.info(f"Starting evaluation on the test set {test_dir}...")
    evaluation_results = evaluator.evaluate_single_model(
        test_dir=str(test_dir),
        output_file=str(output_file)
    )
    
    # Print evaluation summary
    print_evaluation_summary(evaluation_results)
    
    # Visualize results
    if cfg.evaluation.visualize_results:
        visualize_results(evaluation_results, results_dir)
    
    logger.info(f"Evaluation completed, results saved to: {output_file}")
    
    return evaluation_results

def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print evaluation summary
    
    Args:
        results: Evaluation results dictionary
    """
    summary = results['summary']
    model_summary = summary['model']
    
    total_instances = len(results['instances'])
    successful_instances = sum(1 for instance in results['instances'] if instance.get('basis_applied_successfully', False))
    
    # Collect more statistics
    baseline_runtimes = []
    model_runtimes = []
    runtime_improvements = []
    node_improvements = []
    iter_improvements = []
    improvement_counts = 0
    worsening_counts = 0
    no_change_counts = 0
    
    for instance in results['instances']:
        if instance.get('basis_applied_successfully', False):
            baseline = instance.get('baseline', {})
            model = instance.get('model_results', {})
            improvements = instance.get('improvements', {})
            
            # Collect runtime
            baseline_runtime = baseline.get('runtime')
            model_runtime = model.get('runtime')
            if baseline_runtime is not None:
                baseline_runtimes.append(baseline_runtime)
            if model_runtime is not None:
                model_runtimes.append(model_runtime)
            
            # Collect improvement data
            runtime_improvement = improvements.get('runtime_improvement')
            if runtime_improvement is not None:
                runtime_improvements.append(runtime_improvement)
                if runtime_improvement < 0:
                    improvement_counts += 1
                elif runtime_improvement > 0:
                    worsening_counts += 1
                else:
                    no_change_counts += 1
            
            # Collect node count and iteration count improvement
            node_improvement = improvements.get('node_count_improvement')
            if node_improvement is not None:
                node_improvements.append(node_improvement)
                
            iter_improvement = improvements.get('iteration_count_improvement')
            if iter_improvement is not None:
                iter_improvements.append(iter_improvement)
    
    # Calculate statistics
    avg_baseline_runtime = np.mean(baseline_runtimes) if baseline_runtimes else None
    avg_model_runtime = np.mean(model_runtimes) if model_runtimes else None
    avg_runtime_improvement = np.mean(runtime_improvements) if runtime_improvements else None
    median_runtime_improvement = np.median(runtime_improvements) if runtime_improvements else None
    avg_node_improvement = np.mean(node_improvements) if node_improvements else None
    avg_iter_improvement = np.mean(iter_improvements) if iter_improvements else None
    
    # Update the results dictionary to ensure these statistics are included in the JSON
    if 'detailed_summary' not in results['summary']:
        results['summary']['detailed_summary'] = {}
    
    detailed_summary = results['summary']['detailed_summary']
    detailed_summary.update({
        'avg_baseline_runtime': avg_baseline_runtime,
        'avg_model_runtime': avg_model_runtime,
        'avg_runtime_improvement_pct': avg_runtime_improvement * 100 if avg_runtime_improvement is not None else None,
        'median_runtime_improvement_pct': median_runtime_improvement * 100 if median_runtime_improvement is not None else None,
        'avg_node_improvement_pct': avg_node_improvement * 100 if avg_node_improvement is not None else None,
        'avg_iter_improvement_pct': avg_iter_improvement * 100 if avg_iter_improvement is not None else None,
        'instances_with_improvement': improvement_counts,
        'instances_with_worsening': worsening_counts,
        'instances_with_no_change': no_change_counts,
        'improvement_ratio': improvement_counts / successful_instances if successful_instances > 0 else 0,
        'basis_application_ratio': successful_instances / total_instances if total_instances > 0 else 0
    })
    
    # ========== Print evaluation summary ==========
    logger.info(f"\n{'='*80}\nEvaluation summary (Total {total_instances} instances)\n{'='*80}")
    
    # Basic information
    logger.info(f"\nBasic information:")
    logger.info(f"  Total number of instances: {total_instances}")
    logger.info(f"  Number of instances successfully applied initial basis: {successful_instances}/{total_instances} ({successful_instances/total_instances*100:.2f}%)")
    logger.info(f"  Initial basis setting success rate: {model_summary['basis_set_success_count']}/{total_instances} ({model_summary['basis_set_success_count']/total_instances*100:.2f}%)")
    
    # Runtime statistics
    logger.info(f"\nRuntime statistics:")
    if avg_baseline_runtime is not None:
        logger.info(f"  Baseline average runtime: {avg_baseline_runtime:.4f} seconds")
    if avg_model_runtime is not None:
        logger.info(f"  Using initial basis average runtime: {avg_model_runtime:.4f} seconds")
    if avg_runtime_improvement is not None:
        sign = "+" if avg_runtime_improvement > 0 else ""
        logger.info(f"  Average runtime improvement: {sign}{avg_runtime_improvement*100:.2f}% (Negative value indicates improvement)")
    if median_runtime_improvement is not None:
        sign = "+" if median_runtime_improvement > 0 else ""
        logger.info(f"  Median runtime improvement: {sign}{median_runtime_improvement*100:.2f}%")
    
    # Improvement instance analysis
    logger.info(f"\nImprovement instance analysis:")
    if successful_instances > 0:
        logger.info(f"  Instances with runtime improvement: {improvement_counts}/{successful_instances} ({improvement_counts/successful_instances*100:.2f}%)")
        logger.info(f"  Instances with runtime worsening: {worsening_counts}/{successful_instances} ({worsening_counts/successful_instances*100:.2f}%)")
        logger.info(f"  Instances with no runtime change: {no_change_counts}/{successful_instances} ({no_change_counts/successful_instances*100:.2f}%)")
    
    # Other performance metrics
    logger.info(f"\nOther performance metrics:")
    if avg_node_improvement is not None:
        sign = "+" if avg_node_improvement > 0 else ""
        logger.info(f"  Average B&B node count improvement: {sign}{avg_node_improvement*100:.2f}%")
    if avg_iter_improvement is not None:
        sign = "+" if avg_iter_improvement > 0 else ""
        logger.info(f"  Average iteration count improvement: {sign}{avg_iter_improvement*100:.2f}%")
    
    # Overall conclusion
    logger.info("\nOverall conclusion:")
    
    if avg_runtime_improvement is not None:
        if avg_runtime_improvement < 0:
            logger.info(f"  On the runtime aspect, the model provides an average improvement of {abs(avg_runtime_improvement*100):.2f}%")
        else:
            logger.info(f"  On the runtime aspect, the model increases an average of {avg_runtime_improvement*100:.2f}% overhead")
    
    logger.info(f"  In the {total_instances} test instances, the success rate of applying initial basis is {successful_instances/total_instances*100:.2f}%")
    if successful_instances > 0:
        logger.info(f"  In the {successful_instances} instances that successfully applied initial basis, {improvement_counts} ({improvement_counts/successful_instances*100:.2f}%) instances have runtime improvements")
    
    # Model quality assessment
    if avg_runtime_improvement is not None:
        if avg_runtime_improvement < -0.1:
            logger.info("  The predicted initial basis significantly improves the performance of Gurobi, and the model training is successful")
        elif avg_runtime_improvement < 0:
            logger.info("  The predicted initial basis slightly improves the performance of Gurobi, but there is still room for improvement")
        elif avg_runtime_improvement > 0.1:
            logger.info("  The predicted initial basis significantly reduces the performance of Gurobi, and the model may need more training or adjustment")
        else:
            logger.info("  The predicted initial basis has inconsistent or limited impact on the performance of Gurobi")
    
    logger.info(f"\n{'='*80}\n")

def visualize_results(results: Dict[str, Any], output_dir: str):
    """
    Visualize the evaluation results of a single model
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
    """
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Collect runtime improvement data
    runtime_data = []
    
    # Process the results of each instance
    for instance in results['instances']:
        if 'result' in instance and instance.get('success', False):
            # Runtime improvement
            if 'runtime_improvement' in instance['result']:
                imp = instance['result']['runtime_improvement']
                if imp is not None:
                    runtime_data.append({
                        'Instance': instance['instance_name'],
                        'Runtime Improvement (%)': -imp * 100  # The negative sign makes the improvement a positive value
                    })
    
    # Create a DataFrame
    runtime_df = pd.DataFrame(runtime_data) if runtime_data else pd.DataFrame()
    
    # Plot the runtime improvement distribution
    if not runtime_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y='Runtime Improvement (%)', data=runtime_df)
        plt.title('Runtime improvement distribution (%)', fontsize=14)
        plt.ylabel('Improvement percentage (%)', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--')  # Add y=0 reference line
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_improvement_distribution.png"), dpi=300)
        
        # Plot the runtime improvement for each instance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Instance', y='Runtime Improvement (%)', data=runtime_df)
        plt.title('Runtime improvement for each instance (%)', fontsize=14)
        plt.xlabel('Instance', fontsize=12)
        plt.ylabel('Improvement percentage (%)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)  # Add y=0 reference line
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_improvement_by_instance.png"), dpi=300)
    
    # Plot the success rate chart
    summary = results['summary']
    total_instances = len(results['instances'])
    
    success_rate = summary['model']['basis_set_success_count'] / total_instances * 100
    
    plt.figure(figsize=(8, 6))
    plt.bar(['GNN Model'], [success_rate])
    plt.title('Initial basis setting success rate (%)', fontsize=14)
    plt.ylabel('Success rate (%)', fontsize=12)
    plt.ylim(0, 100)  # Set the y-axis range from 0 to 100
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basis_success_rate.png"), dpi=300)
    
    # Close all charts
    plt.close('all')

if __name__ == "__main__":
    main()
