"""
run_priority_experiments.py

Focused experiment runner for the most important BC goal conditioning comparisons.
Prioritized based on your teammate experiment list.
"""

import os
import subprocess
import datetime
import json
import argparse
import time

def run_experiment(config, base_save_dir, data_dir, run_number=1):
    """Run a single experiment with the given configuration"""
    
    # Construct data path based on conditioning type and task
    base_filename = config['data_file']
    
    if config['condition_type'] == 'none' or config.get('no_goal', False):
        # Use raw data for baseline
        data_filename = base_filename
    else:
        # Use post-processed data for goal conditioning
        base_name = base_filename.replace('.npz', '')
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
        else:
            data_filename = base_filename
    
    data_path = os.path.join(data_dir, data_filename)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found: {data_path}")
        return False
    
    # Construct save directory with run number
    exp_name = f"{config['task']}_{config['condition_type']}_{config['name']}_run{run_number}"
    save_dir = os.path.join(base_save_dir, exp_name)
    
    # Construct command
    cmd = [
        'python', 'scripts/train_goal_conditioned_bc.py',
        '--data', data_path,
        '--save_dir', save_dir,
        '--condition_type', config['condition_type'],
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['learning_rate']),
        '--layers', str(config['layers']),
        '--hidden_size', str(config['hidden_size']),
        '--eval_interval', str(config['eval_interval']),
        '--eval_env', config['eval_env'],
        '--device', config.get('device', 'cuda'),
        '--no_timestamp'
    ]
    
    # Add optional flags
    if config.get('no_goal', False):
        cmd.append('--no_goal')
    if config.get('no_augmentation', True):  # Default to no augmentation
        cmd.append('--no_augmentation')
    if config.get('no_curriculum', True):   # Default to no curriculum
        cmd.append('--no_curriculum')
    
    print(f"\n📊 Data file: {data_filename}")
    print(f"🚀 Command: {' '.join(cmd[-10:])}")  # Show last 10 args
    
    start_time = time.time()
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.1f}s!")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Failed after {elapsed_time:.1f}s with return code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run priority BC experiments for teammate comparison')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data files')
    parser.add_argument('--save_dir', type=str, default='./priority_experiments',
                      help='Base directory to save all experiments')
    parser.add_argument('--priority', type=str, 
                      choices=['high', 'medium', 'full', 'test'],
                      default='high',
                      help='Priority level: high=most important, medium=extended, full=all, test=quick validation')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs per experiment (default: 3 for statistical significance)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create base save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(args.save_dir, f"priority_bc_experiments_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Base configs for different tasks
    base_2block_config = {
        'task': 'two_blocks',
        'data_file': 'data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz',
        'batch_size': 64,
        'learning_rate': 5e-4,
        'layers': 3,
        'hidden_size': 128,
        'eval_env': 'PegTransfer-v0',
        'device': args.device
    }
    
    # Define experiment priorities based on your list
    experiments = []
    
    if args.priority == 'test':
        # Quick validation experiments (10 epochs)
        experiments = [
            {**base_2block_config, 'condition_type': 'none', 'name': 'baseline', 'no_goal': True, 'epochs': 10, 'eval_interval': 2},
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot', 'epochs': 10, 'eval_interval': 2},
        ]
        print("🧪 TEST MODE: Quick validation experiments")
        
    elif args.priority == 'high':
        # HIGH PRIORITY: Core 4 goal conditioning methods (3 runs each, 100 epochs)
        experiments = [
            # 2 Block | One hot | All blocks (not colored) - KEY COMPARISON
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_standard', 'epochs': 100, 'eval_interval': 10},
            
            # 2 Block | Target block position | All blocks (not colored) 
            {**base_2block_config, 'condition_type': 'target_block', 'name': 'targetblock_standard', 'epochs': 100, 'eval_interval': 10},
            
            # 2 Block | Target block and goal peg position | All blocks (not colored)
            {**base_2block_config, 'condition_type': 'target_block_and_target_peg', 'name': 'targetblock_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
            
            # 2 Block | One hot and goal peg position | All blocks (not colored)
            {**base_2block_config, 'condition_type': 'one_hot_and_target_peg', 'name': 'onehot_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
        ]
        print("🎯 HIGH PRIORITY: 4 core goal conditioning methods (3 runs × 100 epochs each)")
        
    elif args.priority == 'medium':
        # MEDIUM PRIORITY: Extended comparison including architectural variations
        experiments = [
            # High priority experiments
            {**base_2block_config, 'condition_type': 'none', 'name': 'baseline', 'no_goal': True, 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'target_block', 'name': 'targetblock_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'one_hot_and_target_peg', 'name': 'onehot_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'target_block_and_target_peg', 'name': 'targetblock_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
            
            # Additional comparisons
            {**base_2block_config, 'condition_type': 'four_tuple', 'name': 'fourtuple_standard', 'epochs': 100, 'eval_interval': 10},
            
            # Architecture variations for best performing method
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_deep', 'epochs': 100, 'eval_interval': 10, 'layers': 4, 'hidden_size': 256},
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_wide', 'epochs': 100, 'eval_interval': 10, 'layers': 3, 'hidden_size': 256},
        ]
        print("📈 MEDIUM PRIORITY: 8 experiments including architectural variations")
        
    elif args.priority == 'full':
        # FULL: All experiments from your list that we can run
        base_4block_config = {
            **base_2block_config,
            'task': 'four_blocks',
            'data_file': 'data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz',  # Update if you have 4-block data
        }
        
        experiments = [
            # 2 Block experiments
            {**base_2block_config, 'condition_type': 'none', 'name': 'baseline', 'no_goal': True, 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'target_block', 'name': 'targetblock_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'one_hot_and_target_peg', 'name': 'onehot_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'target_block_and_target_peg', 'name': 'targetblock_targetpeg_standard', 'epochs': 100, 'eval_interval': 10},
            {**base_2block_config, 'condition_type': 'four_tuple', 'name': 'fourtuple_standard', 'epochs': 100, 'eval_interval': 10},
            
            # Architecture variations
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_deep', 'epochs': 100, 'eval_interval': 10, 'layers': 4, 'hidden_size': 256},
            {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_wide', 'epochs': 100, 'eval_interval': 10, 'layers': 3, 'hidden_size': 256},
            
            # 4 Block experiments (if data available)
            # {**base_4block_config, 'condition_type': 'target_block', 'name': 'targetblock_4block', 'epochs': 50, 'eval_interval': 5},
            # {**base_4block_config, 'condition_type': 'target_block_and_target_peg', 'name': 'targetblock_targetpeg_4block', 'epochs': 50, 'eval_interval': 5},
        ]
        print("🚀 FULL PRIORITY: Comprehensive experiment suite")
    
    total_runs = len(experiments) * args.runs
    
    print(f"\n{'='*80}")
    print(f"🎯 BC GOAL CONDITIONING PRIORITY EXPERIMENTS")
    print(f"{'='*80}")
    print(f"📊 Priority level: {args.priority.upper()}")
    print(f"🔢 Experiments: {len(experiments)}")
    print(f"🔄 Runs per experiment: {args.runs}")
    print(f"📈 Total training runs: {total_runs}")
    print(f"💻 Device: {args.device}")
    print(f"📁 Save directory: {base_save_dir}")
    
    # Estimate time
    avg_epochs = sum(exp['epochs'] for exp in experiments) / len(experiments) if experiments else 0
    estimated_time_per_run = avg_epochs * 0.8  # Rough estimate: 0.8 min per epoch
    estimated_total_time = total_runs * estimated_time_per_run
    print(f"⏰ Estimated time: {estimated_total_time:.0f} minutes ({estimated_total_time/60:.1f} hours)")
    
    # Save experiment configuration
    config_path = os.path.join(base_save_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'experiments': experiments,
            'timestamp': timestamp,
            'total_runs': total_runs,
            'estimated_time_minutes': estimated_total_time
        }, f, indent=2)
    
    # Ask for confirmation unless test mode
    if args.priority != 'test':
        response = input(f"\n🚀 Proceed with {total_runs} training runs? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted.")
            return
    
    # Run experiments
    results = []
    start_time = time.time()
    current_exp = 0
    
    for exp_idx, config in enumerate(experiments):
        print(f"\n{'#'*80}")
        print(f"📋 EXPERIMENT {exp_idx+1}/{len(experiments)}: {config['condition_type']}_{config['name']}")
        print(f"⚙️  Config: {config['epochs']} epochs, {config['layers']}x{config['hidden_size']} network")
        print(f"{'#'*80}")
        
        group_successes = 0
        for run in range(1, args.runs + 1):
            current_exp += 1
            print(f"\n🏃 RUN {current_exp}/{total_runs} (Exp {exp_idx+1}, Run {run}/{args.runs})")
            
            success = run_experiment(config, base_save_dir, args.data_dir, run)
            results.append({
                'config': config,
                'run_number': run,
                'success': success,
                'experiment_number': current_exp,
                'config_index': exp_idx,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            if success:
                group_successes += 1
            
            # Save intermediate results
            results_path = os.path.join(base_save_dir, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        success_rate = group_successes / args.runs
        status = "✅" if success_rate == 1.0 else "⚠️ " if success_rate > 0 else "❌"
        print(f"{status} Group completed: {group_successes}/{args.runs} runs successful ({success_rate*100:.0f}%)")
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎯 FINAL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"📊 Overall success: {successful}/{total_runs} runs ({successful/total_runs*100:.1f}%)")
    print(f"⏰ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    # Group by experiment type
    exp_summary = {}
    for result in results:
        config = result['config']
        exp_key = f"{config['condition_type']}_{config['name']}"
        if exp_key not in exp_summary:
            exp_summary[exp_key] = {'success': 0, 'total': 0, 'condition_type': config['condition_type']}
        exp_summary[exp_key]['total'] += 1
        if result['success']:
            exp_summary[exp_key]['success'] += 1
    
    print(f"\n📈 Results by experiment:")
    baseline_success = None
    for exp_name, summary in sorted(exp_summary.items()):
        success_rate = summary['success'] / summary['total']
        status = "✅" if success_rate == 1.0 else "⚠️ " if success_rate > 0 else "❌"
        condition = summary['condition_type']
        print(f"{status} {exp_name:35} ({condition:15}): {summary['success']}/{summary['total']} runs")
        
        if 'baseline' in exp_name:
            baseline_success = success_rate
    
    # Save final results
    final_results = {
        'summary': {
            'priority_level': args.priority,
            'total_runs': total_runs,
            'successful_runs': successful,
            'success_rate': successful / total_runs,
            'total_time_minutes': total_time / 60,
            'experiments_by_type': exp_summary
        },
        'individual_results': results,
        'config': vars(args),
        'timestamp': timestamp
    }
    
    final_results_path = os.path.join(base_save_dir, 'final_results.json')
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📁 All results saved to: {base_save_dir}")
    print(f"📊 Final results: {final_results_path}")
    
    print(f"\n🎉 Priority experiments completed!")
    print(f"🔬 Ready for comparison with teammates' HER+DDPG results!")

if __name__ == '__main__':
    main()