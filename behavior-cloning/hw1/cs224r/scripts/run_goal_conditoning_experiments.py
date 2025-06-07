"""
run_goal_conditioning_experiments.py

Batch runner for goal conditioning experiments to systematically test different approaches.
"""

import os
import subprocess
import datetime
import json
import argparse

def run_experiment(config, base_save_dir, data_dir, run_number=1):
    """Run a single experiment with the given configuration"""
    
    # Construct data path
    data_filename = config['data_file']
    if config['condition_type'] != 'none' and not config.get('no_goal', False):
        # Need postprocessed data
        base_name = data_filename.replace('.npz', '')
        if config['condition_type'] == 'one_hot':
            data_filename = f"{base_name}_onehot.npz"
        elif config['condition_type'] == 'target_block':
            data_filename = f"{base_name}_targetblock.npz"
        elif config['condition_type'] == 'target_block_and_target_peg':
            data_filename = f"{base_name}_targetblocktargetpeg.npz"
        elif config['condition_type'] == 'one_hot_and_target_peg':
            data_filename = f"{base_name}_onehottargetpeg.npz"
        elif config['condition_type'] == 'four_tuple':
            data_filename = f"{base_name}_fourtuple.npz"
    
    data_path = os.path.join(data_dir, data_filename)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print(f"You may need to run data_postprocessing.py first")
        return False
    
    # Construct save directory with run number
    exp_name = f"{config['task']}_{config['condition_type']}_{config['name']}_run{run_number}"
    save_dir = os.path.join(base_save_dir, exp_name)
    
    # Construct command
    cmd = [
        'python', 'train_goal_conditioned_bc.py',
        '--data', data_path,
        '--save_dir', save_dir,
        '--condition_type', config['condition_type'],
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['learning_rate']),
        '--layers', str(config['layers']),
        '--hidden_size', str(config['hidden_size']),
        '--eval_interval', str(config['eval_interval']),
        '--eval_env', config['eval_env']
    ]
    
    # Add optional flags
    if config.get('no_goal', False):
        cmd.append('--no_goal')
    if config.get('no_augmentation', False):
        cmd.append('--no_augmentation')
    if config.get('no_curriculum', False):
        cmd.append('--no_curriculum')
    if config.get('no_timestamp', False):
        cmd.append('--no_timestamp')
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run batch goal conditioning experiments')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data files')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                      help='Base directory to save all experiments')
    parser.add_argument('--task', type=str, choices=['two_blocks', 'four_blocks'], 
                      default='two_blocks',
                      help='Which task variant to run')
    parser.add_argument('--data_file', type=str, 
                      default='data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz',
                      help='Base data file name (without conditioning suffix)')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick experiments (fewer epochs)')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs per experiment (default: 3)')
    parser.add_argument('--run_start', type=int, default=1,
                      help='Starting run number (for resuming experiments)')
    parser.add_argument('--experiments', type=str, nargs='+', 
                      choices=['basic', 'ablation', 'architecture', 'baseline', 'all'],
                      default=['basic'],
                      help='Which experiment groups to run')
    
    args = parser.parse_args()
    
    # Create base save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(args.save_dir, f"goal_conditioning_experiments_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Define experiment configurations
    base_config = {
        'task': args.task,
        'data_file': args.data_file,
        'epochs': 50 if args.quick else 200,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'layers': 3,
        'hidden_size': 128,
        'eval_interval': 5 if args.quick else 10,
        'eval_env': 'PegTransfer-v0',
        'no_timestamp': True
    }
    
    experiments = []
    
    # Baseline experiments (no goal conditioning)
    if 'baseline' in args.experiments or 'all' in args.experiments:
        baseline_experiments = [
            # Raw data, no goal conditioning
            {**base_config, 'condition_type': 'one_hot', 'name': 'no_goal_baseline', 'no_goal': True},
        ]
        experiments.extend(baseline_experiments)
    
    # Basic conditioning experiments
    if 'basic' in args.experiments or 'all' in args.experiments:
        basic_experiments = [
            {**base_config, 'condition_type': 'one_hot', 'name': 'basic'},
            {**base_config, 'condition_type': 'target_block', 'name': 'basic'},
            {**base_config, 'condition_type': 'one_hot_and_target_peg', 'name': 'basic'},
            {**base_config, 'condition_type': 'four_tuple', 'name': 'basic'},
        ]
        experiments.extend(basic_experiments)
    
    # Ablation studies
    if 'ablation' in args.experiments or 'all' in args.experiments:
        ablation_experiments = [
            # No goal conditioning baseline
            {**base_config, 'condition_type': 'one_hot', 'name': 'no_goal', 'no_goal': True},
            # No data augmentation
            {**base_config, 'condition_type': 'one_hot', 'name': 'no_augmentation', 'no_augmentation': True},
            # No curriculum learning
            {**base_config, 'condition_type': 'one_hot', 'name': 'no_curriculum', 'no_curriculum': True},
        ]
        experiments.extend(ablation_experiments)
    
    # Architecture experiments
    if 'architecture' in args.experiments or 'all' in args.experiments:
        architecture_experiments = [
            # Deeper network
            {**base_config, 'condition_type': 'one_hot', 'name': 'deep', 'layers': 4, 'hidden_size': 256},
            # Wider network
            {**base_config, 'condition_type': 'one_hot', 'name': 'wide', 'layers': 3, 'hidden_size': 256},
            # Smaller network
            {**base_config, 'condition_type': 'one_hot', 'name': 'small', 'layers': 2, 'hidden_size': 64},
        ]
        experiments.extend(architecture_experiments)
    
    print(f"Planning to run {len(experiments)} experiments x {args.runs} runs = {len(experiments) * args.runs} total")
    print(f"Base save directory: {base_save_dir}")
    
    # Save experiment configuration
    config_path = os.path.join(base_save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'experiments': experiments,
            'timestamp': timestamp,
            'total_runs': len(experiments) * args.runs
        }, f, indent=2)
    
    # Run experiments - multiple runs per experiment
    results = []
    total_experiments = len(experiments) * args.runs
    current_exp = 0
    
    for exp_idx, config in enumerate(experiments):
        for run in range(args.run_start, args.runs + 1):
            current_exp += 1
            print(f"\n{'#'*80}")
            print(f"EXPERIMENT {current_exp}/{total_experiments}")
            print(f"Config: {config['task']}_{config['condition_type']}_{config['name']}")
            print(f"Run: {run}/{args.runs}")
            print(f"{'#'*80}")
            
            success = run_experiment(config, base_save_dir, args.data_dir, run)
            results.append({
                'config': config,
                'run_number': run,
                'success': success,
                'experiment_number': current_exp,
                'config_index': exp_idx
            })
        
        # Save intermediate results
        results_path = os.path.join(base_save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Completed: {successful}/{total_experiments} total runs")
    
    # Group by experiment type
    exp_summary = {}
    for result in results:
        config = result['config']
        exp_key = f"{config['task']}_{config['condition_type']}_{config['name']}"
        if exp_key not in exp_summary:
            exp_summary[exp_key] = {'success': 0, 'total': 0}
        exp_summary[exp_key]['total'] += 1
        if result['success']:
            exp_summary[exp_key]['success'] += 1
    
    for exp_name, summary in exp_summary.items():
        success_rate = summary['success'] / summary['total']
        status = "✓" if success_rate == 1.0 else "⚠" if success_rate > 0 else "✗"
        print(f"{status} {exp_name}: {summary['success']}/{summary['total']} runs successful")
    
    print(f"\nAll results saved to: {base_save_dir}")
    print(f"Experiment config: {config_path}")
    print(f"Results summary: {results_path}")

if __name__ == '__main__':
    main()