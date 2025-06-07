"""
Comprehensive Goal Conditioning Experiments Runner

This script runs all goal conditioning experiments with proper dimension handling:
1. Baseline (none)
2. One-hot encoding  
3. Target block position
4. Target block + target peg positions
5. One-hot + target peg position

Each experiment uses the correct observation dimensions and model architecture.
"""

import os
import subprocess
import datetime
import json
import argparse
import time
from typing import List, Dict

def create_experiment_config():
    """Create configuration for all goal conditioning experiments"""
    
    base_config = {
        'epochs': 100,
        'batch_size': 64,
        'eval_interval': 10,
        'data_path': '/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz'
    }
    
    experiments = [
        {
            'name': 'baseline',
            'method': 'none',
            'description': 'Baseline with no goal conditioning',
            'expected_input_dim': 13,  # Base observation only
            **base_config
        },
        {
            'name': 'one_hot_encoding',
            'method': 'one_hot',
            'description': 'Block identity as one-hot vector',
            'expected_input_dim': 15,  # 13 + 2 (block encoding)
            **base_config
        },
        {
            'name': 'target_block_position',
            'method': 'target_block',
            'description': 'Target block 3D position',
            'expected_input_dim': 16,  # 13 + 3 (achieved goal)
            **base_config
        },
        {
            'name': 'full_spatial_conditioning',
            'method': 'target_block_and_target_peg',
            'description': 'Target block + target peg positions',
            'expected_input_dim': 19,  # 13 + 3 + 3 (achieved + desired goals)
            **base_config
        },
        {
            'name': 'hybrid_conditioning',
            'method': 'one_hot_and_target_peg',
            'description': 'Block identity + target peg position',
            'expected_input_dim': 18,  # 13 + 2 + 3 (block encoding + desired goal)
            **base_config
        }
    ]
    
    return experiments

def run_single_experiment(experiment: Dict, base_save_dir: str, run_id: int = 1) -> Dict:
    """Run a single goal conditioning experiment"""
    
    # Create experiment-specific save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_save_dir = os.path.join(base_save_dir, f"{experiment['name']}_run{run_id}_{timestamp}")
    
    print(f"\n🧪 RUNNING EXPERIMENT: {experiment['name'].upper()}")
    print(f"="*60)
    print(f"   Method: {experiment['method']}")
    print(f"   Description: {experiment['description']}")
    print(f"   Expected Input Dim: {experiment['expected_input_dim']}")
    print(f"   Save Dir: {exp_save_dir}")
    
    # Construct command
    cmd = [
        'python', 'universal_bc_trainer.py',
        '--data_path', experiment['data_path'],
        '--method', experiment['method'],
        '--save_dir', exp_save_dir,
        '--epochs', str(experiment['epochs']),
        '--batch_size', str(experiment['batch_size']),
        '--eval_interval', str(experiment['eval_interval'])
    ]
    
    print(f"   Command: {' '.join(cmd[-6:])}")  # Show key arguments
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        # Try to load results
        results_path = os.path.join(exp_save_dir, f'results_{experiment["method"]}.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                exp_results = json.load(f)
            
            success_rate = exp_results.get('best_success_rate', 0.0)
            
            print(f"   ✅ SUCCESS! Time: {elapsed_time:.1f}s, Success Rate: {success_rate:.1%}")
            
            return {
                'experiment': experiment['name'],
                'method': experiment['method'],
                'success': True,
                'success_rate': success_rate,
                'elapsed_time': elapsed_time,
                'save_dir': exp_save_dir,
                'results_file': results_path
            }
        else:
            print(f"   ⚠️  Completed but no results file found")
            return {
                'experiment': experiment['name'],
                'method': experiment['method'],
                'success': True,
                'success_rate': 0.0,
                'elapsed_time': elapsed_time,
                'save_dir': exp_save_dir,
                'error': 'No results file found'
            }
    
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print(f"   ❌ FAILED! Time: {elapsed_time:.1f}s, Error Code: {e.returncode}")
        print(f"   Error Output: {e.stderr[:200]}...")
        
        return {
            'experiment': experiment['name'],
            'method': experiment['method'],
            'success': False,
            'success_rate': 0.0,
            'elapsed_time': elapsed_time,
            'error': e.stderr[:500],
            'return_code': e.returncode
        }

def run_evaluation_suite(experiment_results: List[Dict], base_save_dir: str):
    """Run evaluation on all successful experiments"""
    print(f"\n🎯 RUNNING EVALUATION SUITE")
    print("="*80)
    
    successful_experiments = [exp for exp in experiment_results if exp['success'] and exp.get('save_dir')]
    
    if not successful_experiments:
        print("❌ No successful experiments to evaluate")
        return {}
    
    print(f"Evaluating {len(successful_experiments)} successful experiments...")
    
    evaluation_results = {}
    
    for exp in successful_experiments:
        try:
            print(f"\n📊 Evaluating {exp['experiment']} ({exp['method']})...")
            
            # Run evaluation
            cmd = [
                'python', 'universal_bc_evaluator.py',
                '--model_dir', exp['save_dir'],
                '--method', exp['method'],
                '--num_episodes', '20'
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load evaluation results
            eval_results_path = os.path.join(exp['save_dir'], f'evaluation_results_{exp["method"]}.json')
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'r') as f:
                    eval_data = json.load(f)
                
                evaluation_results[exp['method']] = eval_data
                print(f"   ✅ Success Rate: {eval_data['success_rate']:.1%}, Mean Return: {eval_data['mean_return']:.2f}")
            else:
                print(f"   ⚠️  Evaluation completed but no results file found")
        
        except Exception as e:
            print(f"   ❌ Evaluation failed: {str(e)[:100]}...")
            evaluation_results[exp['method']] = {'error': str(e), 'success_rate': 0.0}
    
    # Save combined evaluation results
    eval_summary_path = os.path.join(base_save_dir, 'evaluation_summary.json')
    with open(eval_summary_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n💾 Evaluation summary saved to: {eval_summary_path}")
    
    return evaluation_results

def create_summary_report(experiment_results: List[Dict], evaluation_results: Dict, save_path: str):
    """Create comprehensive summary report"""
    print(f"\n📋 CREATING SUMMARY REPORT")
    print("="*60)
    
    # Training summary
    training_summary = {
        'total_experiments': len(experiment_results),
        'successful_experiments': sum(1 for exp in experiment_results if exp['success']),
        'failed_experiments': sum(1 for exp in experiment_results if not exp['success']),
        'total_time': sum(exp['elapsed_time'] for exp in experiment_results),
        'experiments': experiment_results
    }
    
    # Evaluation summary
    eval_summary = {}
    for method, eval_data in evaluation_results.items():
        if 'error' not in eval_data:
            eval_summary[method] = {
                'success_rate': eval_data.get('success_rate', 0.0),
                'mean_return': eval_data.get('mean_return', 0.0),
                'episodes_evaluated': eval_data.get('total_episodes', 0)
            }
        else:
            eval_summary[method] = {'error': eval_data['error']}
    
    # Combined report
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'training_summary': training_summary,
        'evaluation_summary': eval_summary,
        'methodology': {
            'goal_conditioning_methods': [
                'none (baseline)',
                'one_hot (block identity)',
                'target_block (spatial)',
                'target_block_and_target_peg (full spatial)',
                'one_hot_and_target_peg (hybrid)'
            ],
            'training_epochs': 100,
            'evaluation_episodes': 20,
            'data_source': '1000 PegTransfer demonstrations'
        }
    }
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary table
    print(f"\n📊 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Training':<10} {'Success Rate':<12} {'Mean Return':<12}")
    print("-"*80)
    
    for exp in experiment_results:
        method = exp['method']
        training_status = "✅ Pass" if exp['success'] else "❌ Fail"
        
        if method in eval_summary and 'error' not in eval_summary[method]:
            success_rate = f"{eval_summary[method]['success_rate']:.1%}"
            mean_return = f"{eval_summary[method]['mean_return']:.2f}"
        else:
            success_rate = "N/A"
            mean_return = "N/A"
        
        print(f"{method:<25} {training_status:<10} {success_rate:<12} {mean_return:<12}")
    
    print(f"\n💾 Full report saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive goal conditioning experiments')
    parser.add_argument('--save_dir', type=str, default='experiments/goal_conditioning_suite',
                      help='Base directory for all experiments')
    parser.add_argument('--data_path', type=str, 
                      default='/home/ubuntu/project/SurRoL/surrol/data/two_blocks/data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz',
                      help='Path to demonstration data')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs for each experiment')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run evaluation')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation and only run training')
    parser.add_argument('--methods', type=str, nargs='+', 
                      choices=['none', 'one_hot', 'target_block', 'target_block_and_target_peg', 'one_hot_and_target_peg'],
                      help='Specific methods to run (default: all)')
    
    args = parser.parse_args()
    
    # Create base save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"{args.save_dir}_{timestamp}"
    os.makedirs(base_save_dir, exist_ok=True)
    
    print(f"🎯 COMPREHENSIVE GOAL CONDITIONING EXPERIMENTS")
    print("="*80)
    print(f"Base Save Dir: {base_save_dir}")
    print(f"Data Path: {args.data_path}")
    print(f"Training Epochs: {args.epochs}")
    
    # Get experiment configurations
    all_experiments = create_experiment_config()
    
    # Filter experiments if specific methods requested
    if args.methods:
        experiments = [exp for exp in all_experiments if exp['method'] in args.methods]
    else:
        experiments = all_experiments
    
    # Update data path and epochs
    for exp in experiments:
        exp['data_path'] = args.data_path
        exp['epochs'] = args.epochs
    
    print(f"Running {len(experiments)} experiments: {[exp['method'] for exp in experiments]}")
    
    experiment_results = []
    
    # Run training experiments
    if not args.skip_training:
        print(f"\n🚀 STARTING TRAINING PHASE")
        print("="*80)
        
        total_start_time = time.time()
        
        for i, experiment in enumerate(experiments):
            print(f"\nExperiment {i+1}/{len(experiments)}")
            result = run_single_experiment(experiment, base_save_dir)
            experiment_results.append(result)
        
        total_training_time = time.time() - total_start_time
        print(f"\n⏰ Total training time: {total_training_time/60:.1f} minutes")
        
        # Save intermediate results
        training_results_path = os.path.join(base_save_dir, 'training_results.json')
        with open(training_results_path, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
    
    # Run evaluation phase
    evaluation_results = {}
    if not args.skip_evaluation:
        if args.skip_training:
            # Load previous training results
            training_results_path = os.path.join(base_save_dir, 'training_results.json')
            if os.path.exists(training_results_path):
                with open(training_results_path, 'r') as f:
                    experiment_results = json.load(f)
            else:
                print("❌ No training results found for evaluation")
                return
        
        evaluation_results = run_evaluation_suite(experiment_results, base_save_dir)
    
    # Create final report
    if experiment_results:
        report_path = os.path.join(base_save_dir, 'final_report.json')
        create_summary_report(experiment_results, evaluation_results, report_path)
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"📁 Results directory: {base_save_dir}")

if __name__ == '__main__':
    main()