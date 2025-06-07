"""
resume_priority_experiments.py

Resume the priority experiments from where they left off.
Based on the logs, we need to start from experiment 2, run 2.
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
    exp_name = f"{config['task']}_{config['condition_type']}_{config['name']}_run{run_number}_{config['condition_type']}"
    save_dir = os.path.join(base_save_dir, exp_name)
    
    # Ensure the parent directory exists
    os.makedirs(base_save_dir, exist_ok=True)
    
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
    parser = argparse.ArgumentParser(description='Resume priority BC experiments')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data files')
    parser.add_argument('--base_save_dir', type=str, required=True,
                      help='Base directory where experiments are being saved (e.g., cs224r/experiments/priority_bc_experiments_20250602_221018)')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Use the existing base save directory
    base_save_dir = args.base_save_dir
    
    # Ensure the base directory exists
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Base config for the remaining experiments
    base_2block_config = {
        'task': 'two_blocks',
        'data_file': 'data_PegTransfer-v0_random_1000_2025-06-01_10-17-43.npz',
        'batch_size': 64,
        'learning_rate': 5e-4,
        'layers': 3,
        'hidden_size': 128,
        'eval_env': 'PegTransfer-v0',
        'device': args.device,
        'epochs': 100,
        'eval_interval': 10
    }
    
    # Define the remaining experiments from your high priority list
    # Based on your logs, experiment 1 (target_block) is complete
    # We need to resume from experiment 2 (target_block_and_target_peg), run 2
    experiments = [
        # Experiment 2: Target block and goal peg position - need runs 2 and 3
        {**base_2block_config, 'condition_type': 'target_block_and_target_peg', 'name': 'targetblock_targetpeg_standard'},
        
        # Experiment 3: One hot and goal peg position - need all 3 runs
        {**base_2block_config, 'condition_type': 'one_hot_and_target_peg', 'name': 'onehot_targetpeg_standard'},
        
        # Experiment 4: One hot - need all 3 runs  
        {**base_2block_config, 'condition_type': 'one_hot', 'name': 'onehot_standard'},
    ]
    
    print(f"\n{'='*80}")
    print(f"🔄 RESUMING BC GOAL CONDITIONING EXPERIMENTS")
    print(f"{'='*80}")
    print(f"📁 Base save directory: {base_save_dir}")
    print(f"💻 Device: {args.device}")
    print(f"📊 Remaining experiments: {len(experiments)}")
    
    # Define what runs we need for each experiment
    experiment_runs = [
        [2, 3],     # target_block_and_target_peg: need runs 2, 3
        [1, 2, 3],  # one_hot_and_target_peg: need all runs
        [1, 2, 3],  # one_hot: need all runs
    ]
    
    total_remaining_runs = sum(len(runs) for runs in experiment_runs)
    print(f"🔄 Total remaining runs: {total_remaining_runs}")
    
    response = input(f"\n🚀 Resume with {total_remaining_runs} training runs? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Aborted.")
        return
    
    # Run experiments
    results = []
    start_time = time.time()
    current_run = 0
    
    for exp_idx, (config, runs_needed) in enumerate(zip(experiments, experiment_runs)):
        print(f"\n{'#'*80}")
        print(f"📋 EXPERIMENT {exp_idx+2}/{len(experiments)+1}: {config['condition_type']}_{config['name']}")
        print(f"⚙️  Config: {config['epochs']} epochs, {config['layers']}x{config['hidden_size']} network")
        print(f"🔄 Runs needed: {runs_needed}")
        print(f"{'#'*80}")
        
        for run in runs_needed:
            current_run += 1
            print(f"\n🏃 RUN {current_run}/{total_remaining_runs} (Exp {exp_idx+2}, Run {run}/3)")
            
            success = run_experiment(config, base_save_dir, args.data_dir, run)
            results.append({
                'config': config,
                'run_number': run,
                'success': success,
                'experiment_number': exp_idx + 2,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            if not success:
                print(f"⚠️  Run {run} failed, but continuing with remaining runs...")
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("🎯 RESUME COMPLETION SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    print(f"📊 Completed runs: {successful}/{total_remaining_runs} successful ({successful/total_remaining_runs*100:.1f}%)")
    print(f"⏰ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    # Save results
    results_path = os.path.join(base_save_dir, 'resume_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'resume_summary': {
                'total_runs': total_remaining_runs,
                'successful_runs': successful,
                'success_rate': successful / total_remaining_runs,
                'total_time_minutes': total_time / 60,
            },
            'individual_results': results,
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📁 Resume results saved to: {results_path}")
    print(f"🎉 Resume completed!")

if __name__ == '__main__':
    main()