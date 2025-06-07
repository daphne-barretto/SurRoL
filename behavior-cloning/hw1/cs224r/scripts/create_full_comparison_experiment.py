"""
Full Comparison Experiment Creator

This script automates the complete BC goal conditioning pipeline:
1. Takes raw data file
2. Creates all necessary post-processed versions
3. Trains BC models on each version
4. Evaluates and compares results

Compatible with your actual data pipeline.
"""

import os
import subprocess
import datetime
import json
import argparse
from typing import List, Dict

def run_data_postprocessing(base_data_path: str, output_dir: str) -> Dict[str, str]:
    """Run data postprocessing to create all conditioning variants"""
    print(f"🔧 RUNNING DATA POSTPROCESSING")
    print("="*50)
    
    base_name = os.path.splitext(os.path.basename(base_data_path))[0]
    
    # Define conditioning methods and their postprocessing commands
    conditioning_methods = {
        'baseline': {
            'description': 'No conditioning (use raw data)',
            'output_file': base_data_path,  # Use original file
            'needs_postprocessing': False
        },
        'one_hot': {
            'description': 'One-hot block encoding',
            'output_file': os.path.join(output_dir, f'{base_name}_onehot.npz'),
            'needs_postprocessing': True,
            'condition_type': 'one_hot'
        },
        'target_block': {
            'description': 'Target block position',
            'output_file': os.path.join(output_dir, f'{base_name}_targetblock.npz'),
            'needs_postprocessing': True,
            'condition_type': 'target_block'
        },
        'target_block_and_target_peg': {
            'description': 'Target block + target peg positions',
            'output_file': os.path.join(output_dir, f'{base_name}_targetblocktargetpeg.npz'),
            'needs_postprocessing': True,
            'condition_type': 'target_block_target_peg'
        },
        'one_hot_and_target_peg': {
            'description': 'One-hot + target peg position',
            'output_file': os.path.join(output_dir, f'{base_name}_onehottargetpeg.npz'),
            'needs_postprocessing': True,
            'condition_type': 'one_hot_target_peg'
        }
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = {}
    
    for method_name, method_info in conditioning_methods.items():
        print(f"\n📊 Processing {method_name}...")
        print(f"   Description: {method_info['description']}")
        
        if not method_info['needs_postprocessing']:
            # Use original file for baseline
            processed_files[method_name] = method_info['output_file']
            print(f"   ✅ Using original file: {method_info['output_file']}")
        else:
            # Run postprocessing
            output_file = method_info['output_file']
            condition_type = method_info['condition_type']
            
            if os.path.exists(output_file):
                print(f"   ⏭️  File already exists: {output_file}")
                processed_files[method_name] = output_file
                continue
            
            # Construct postprocessing command
            cmd = [
                'python', 'data_postprocessing.py',
                base_data_path,
                '--output_path', output_file,
                '--condition_type', condition_type
            ]
            
            try:
                print(f"   🔄 Running: {' '.join(cmd[-4:])}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                processed_files[method_name] = output_file
                print(f"   ✅ Created: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed: {e.stderr[:100]}...")
                print(f"   ⚠️  Skipping {method_name}")
    
    print(f"\n📋 POSTPROCESSING SUMMARY:")
    for method, file_path in processed_files.items():
        print(f"   • {method:25} → {os.path.basename(file_path)}")
    
    return processed_files

def run_bc_training_experiments(processed_files: Dict[str, str], output_dir: str, 
                               epochs: int = 100, batch_size: int = 64) -> Dict[str, Dict]:
    """Run BC training on all processed data files"""
    print(f"\n🚀 RUNNING BC TRAINING EXPERIMENTS")
    print("="*60)
    
    training_results = {}
    
    for method_name, data_file in processed_files.items():
        print(f"\n🧪 Training {method_name.upper()}...")
        
        # Create experiment directory
        exp_dir = os.path.join(output_dir, f'bc_{method_name}')
        
        # Construct training command
        cmd = [
            'python', 'updated_universal_bc_trainer.py',
            '--data_path', data_file,
            '--save_dir', exp_dir,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--eval_interval', '10'
        ]
        
        start_time = datetime.datetime.now()
        
        try:
            print(f"   Command: {' '.join(cmd[-8:])}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Try to load results
            results_file = os.path.join(exp_dir, 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    exp_results = json.load(f)
                
                best_success = exp_results.get('best_success_rate', 0.0)
                
                training_results[method_name] = {
                    'success': True,
                    'success_rate': best_success,
                    'elapsed_time': elapsed_time,
                    'exp_dir': exp_dir,
                    'data_file': data_file
                }
                
                print(f"   ✅ SUCCESS! Time: {elapsed_time:.1f}s, Success Rate: {best_success:.1%}")
            else:
                training_results[method_name] = {
                    'success': True,
                    'success_rate': 0.0,
                    'elapsed_time': elapsed_time,
                    'exp_dir': exp_dir,
                    'error': 'No results file found'
                }
                print(f"   ⚠️  Completed but no results found")
        
        except subprocess.CalledProcessError as e:
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            
            training_results[method_name] = {
                'success': False,
                'success_rate': 0.0,
                'elapsed_time': elapsed_time,
                'error': e.stderr[:200] if e.stderr else 'Unknown error',
                'return_code': e.returncode
            }
            
            print(f"   ❌ FAILED! Time: {elapsed_time:.1f}s, Error: {e.returncode}")
            print(f"   Error: {e.stderr[:100]}...")
    
    return training_results

def create_comprehensive_evaluation(training_results: Dict[str, Dict], output_dir: str):
    """Create comprehensive evaluation and comparison"""
    print(f"\n📊 CREATING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Filter successful experiments
    successful_experiments = {
        method: result for method, result in training_results.items() 
        if result['success'] and result.get('exp_dir')
    }
    
    if not successful_experiments:
        print("❌ No successful experiments to evaluate")
        return
    
    print(f"Evaluating {len(successful_experiments)} successful experiments...")
    
    evaluation_results = {}
    
    for method, result in successful_experiments.items():
        try:
            print(f"\n📈 Evaluating {method}...")
            
            # Run detailed evaluation
            cmd = [
                'python', 'updated_universal_bc_evaluator.py',
                '--model_dir', result['exp_dir'],
                '--num_episodes', '20'
            ]
            
            eval_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load evaluation results
            eval_file = os.path.join(result['exp_dir'], 'evaluation_results.json')
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                evaluation_results[method] = eval_data
                success_rate = eval_data.get('success_rate', 0.0)
                mean_return = eval_data.get('mean_return', 0.0)
                print(f"   ✅ Success: {success_rate:.1%}, Return: {mean_return:.2f}")
            else:
                print(f"   ⚠️  Evaluation completed but no results file")
        
        except Exception as e:
            print(f"   ❌ Evaluation failed: {str(e)[:50]}...")
            evaluation_results[method] = {'error': str(e)}
    
    # Create final comparison report
    create_final_report(training_results, evaluation_results, output_dir)

def create_final_report(training_results: Dict[str, Dict], evaluation_results: Dict[str, Dict], 
                       output_dir: str):
    """Create final comparison report"""
    print(f"\n📋 CREATING FINAL REPORT")
    print("="*50)
    
    # Combine results
    final_report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'summary': {
            'total_methods': len(training_results),
            'successful_training': sum(1 for r in training_results.values() if r['success']),
            'successful_evaluation': len(evaluation_results),
            'total_time': sum(r.get('elapsed_time', 0) for r in training_results.values())
        },
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'methodology': {
            'description': 'Comprehensive BC goal conditioning comparison',
            'methods_tested': list(training_results.keys()),
            'evaluation_episodes': 20
        }
    }
    
    # Save report
    report_file = os.path.join(output_dir, 'final_comparison_report.json')
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Create summary table
    print(f"\n🏆 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Training':<10} {'Success Rate':<12} {'Mean Return':<12} {'Status'}")
    print("-"*80)
    
    for method in training_results.keys():
        # Training status
        training_status = "✅ Pass" if training_results[method]['success'] else "❌ Fail"
        
        # Evaluation results
        if method in evaluation_results and 'error' not in evaluation_results[method]:
            eval_data = evaluation_results[method]
            success_rate = f"{eval_data.get('success_rate', 0):.1%}"
            mean_return = f"{eval_data.get('mean_return', 0):.2f}"
            status = "🎯 Evaluated"
        elif method in evaluation_results:
            success_rate = "Error"
            mean_return = "Error"
            status = "❌ Eval Failed"
        else:
            success_rate = "N/A"
            mean_return = "N/A"
            status = "⏭️ Skipped"
        
        print(f"{method:<25} {training_status:<10} {success_rate:<12} {mean_return:<12} {status}")
    
    print(f"\n💾 Full report saved to: {report_file}")
    print(f"📁 All results in: {output_dir}")
    
    # Print recommendations
    if evaluation_results:
        best_method = max(evaluation_results.keys(), 
                         key=lambda k: evaluation_results[k].get('success_rate', 0) 
                         if 'error' not in evaluation_results[k] else 0)
        best_success = evaluation_results[best_method].get('success_rate', 0)
        
        print(f"\n🏆 BEST PERFORMING METHOD:")
        print(f"   • {best_method}: {best_success:.1%} success rate")
        
        if best_success < 0.3:
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   • Success rates are low - consider more epochs (200-500)")
            print(f"   • Try different hyperparameters (learning rate, batch size)")
            print(f"   • Ensure demonstration quality is high")
            print(f"   • Consider using more demonstration data")

def main():
    parser = argparse.ArgumentParser(description='Create full BC goal conditioning comparison')
    parser.add_argument('--base_data', type=str, required=True, 
                      help='Path to base data file (e.g., goal_conditioned_demos_with_all_blocks_2_blocks.npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for all experiments')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--skip_postprocessing', action='store_true', 
                      help='Skip postprocessing (files already exist)')
    parser.add_argument('--skip_training', action='store_true', 
                      help='Skip training (models already trained)')
    parser.add_argument('--only_evaluate', action='store_true',
                      help='Only run evaluation on existing models')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"🎯 FULL BC GOAL CONDITIONING COMPARISON")
    print("="*80)
    print(f"Base Data: {args.base_data}")
    print(f"Output Dir: {full_output_dir}")
    print(f"Epochs: {args.epochs}")
    
    # Phase 1: Data postprocessing
    if not args.skip_postprocessing and not args.only_evaluate:
        processed_files = run_data_postprocessing(args.base_data, full_output_dir)
    else:
        print("⏭️  Skipping postprocessing")
        # Load existing processed files
        processed_files = {}  # Would need to implement discovery logic
    
    # Phase 2: BC training
    if not args.skip_training and not args.only_evaluate:
        training_results = run_bc_training_experiments(
            processed_files, full_output_dir, args.epochs, args.batch_size
        )
    else:
        print("⏭️  Skipping training")
        # Load existing training results
        training_results = {}  # Would need to implement discovery logic
    
    # Phase 3: Comprehensive evaluation
    if args.only_evaluate or (not args.skip_training):
        create_comprehensive_evaluation(training_results, full_output_dir)
    
    print(f"\n🎉 FULL COMPARISON COMPLETED!")
    print(f"📁 Results directory: {full_output_dir}")

if __name__ == '__main__':
    main()