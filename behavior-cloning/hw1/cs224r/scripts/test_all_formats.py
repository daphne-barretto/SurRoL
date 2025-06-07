#!/usr/bin/env python3
"""
🧪 Test All Goal Conditioning Formats

Comprehensive testing script to verify compatibility of all goal conditioning formats
with training, evaluation, and visualization scripts.

Usage:
    python test_all_formats.py --data_path ../data/data_PegTransfer-v0_random_1000_2025-06-01_10-06-51.npz
    python test_all_formats.py --quick_test  # Just run format validation
"""

import argparse
import os
import sys
import tempfile
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goal_conditioning_compatibility import (
    GoalConditioningManager, 
    print_format_explanation
)

def test_format_validation(data_path: str) -> dict:
    """Test data validation for all formats"""
    print(f"\n🔍 TESTING FORMAT VALIDATION")
    print(f"="*60)
    
    results = {}
    
    for format_type in GoalConditioningManager.SUPPORTED_FORMATS:
        print(f"\n📋 Testing: {format_type}")
        
        try:
            validation = GoalConditioningManager.validate_conditioning_compatibility(
                data_path, format_type
            )
            
            status = "✅ PASS" if validation['compatible'] else "❌ FAIL"
            print(f"   Status: {status}")
            
            if validation['compatible'] and format_type != 'none':
                print(f"   Dimensions: {validation['base_obs_dim']} + {validation['conditioning_dim']} = {validation['total_obs_dim']}")
            
            if validation['issues']:
                for issue in validation['issues']:
                    print(f"   Issue: {issue}")
            
            results[format_type] = {
                'compatible': validation['compatible'],
                'total_obs_dim': validation.get('total_obs_dim', 0),
                'issues': validation['issues']
            }
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results[format_type] = {
                'compatible': False,
                'error': str(e)
            }
    
    return results

def test_training_compatibility(data_path: str, test_dir: str) -> dict:
    """Test training with different formats (quick training runs)"""
    print(f"\n🏋️ TESTING TRAINING COMPATIBILITY")
    print(f"="*60)
    
    results = {}
    
    # Test a subset of formats for training (full test would take too long)
    test_formats = ['none', 'one_hot', 'target_block']
    
    for format_type in test_formats:
        print(f"\n🎯 Training with format: {format_type}")
        
        save_dir = os.path.join(test_dir, f"test_{format_type}")
        
        try:
            # Construct training command
            cmd = [
                'python', 'train_goal_conditioned_bc.py',
                '--data', data_path,
                '--save_dir', save_dir,
                '--condition_type', format_type,
                '--epochs', '2',  # Just 2 epochs for quick test
                '--batch_size', '32',
                '--eval_episodes', '2',
                '--no_timestamp'
            ]
            
            if format_type == 'none':
                cmd.append('--no_goal')
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(
                cmd, 
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Training successful")
                
                # Check if model was saved
                model_path = os.path.join(save_dir, f'final_model_{format_type}.pt')
                if os.path.exists(model_path):
                    print(f"   ✅ Model saved: {model_path}")
                    results[format_type] = {
                        'training_success': True,
                        'model_path': model_path
                    }
                else:
                    print(f"   ⚠️  Training completed but model not found")
                    results[format_type] = {
                        'training_success': False,
                        'error': 'Model file not found'
                    }
            else:
                print(f"   ❌ Training failed")
                print(f"   Error: {result.stderr}")
                results[format_type] = {
                    'training_success': False,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Training timed out")
            results[format_type] = {
                'training_success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"   ❌ Training error: {e}")
            results[format_type] = {
                'training_success': False,
                'error': str(e)
            }
    
    return results

def test_evaluation_compatibility(training_results: dict, test_dir: str) -> dict:
    """Test evaluation with different formats"""
    print(f"\n🎯 TESTING EVALUATION COMPATIBILITY")
    print(f"="*60)
    
    results = {}
    
    for format_type, train_result in training_results.items():
        if not train_result.get('training_success'):
            print(f"\n⏭️  Skipping evaluation for {format_type} (training failed)")
            continue
        
        print(f"\n📊 Evaluating format: {format_type}")
        
        model_path = train_result['model_path']
        
        try:
            # Test with universal evaluator
            cmd = [
                'python', 'universal_bc_evaluator.py',
                '--model_path', model_path,
                '--num_episodes', '3',  # Quick test
                '--output_json', os.path.join(test_dir, f'eval_{format_type}.json')
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Evaluation successful")
                
                # Load evaluation results
                eval_file = os.path.join(test_dir, f'eval_{format_type}.json')
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    success_rate = eval_data.get('success_rate', 0)
                    print(f"   📊 Success rate: {success_rate:.1%}")
                    
                    results[format_type] = {
                        'evaluation_success': True,
                        'success_rate': success_rate,
                        'compatibility_check': eval_data.get('compatibility_check', True)
                    }
                else:
                    results[format_type] = {
                        'evaluation_success': False,
                        'error': 'Evaluation results not found'
                    }
            else:
                print(f"   ❌ Evaluation failed")
                print(f"   Error: {result.stderr}")
                results[format_type] = {
                    'evaluation_success': False,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Evaluation timed out")
            results[format_type] = {
                'evaluation_success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"   ❌ Evaluation error: {e}")
            results[format_type] = {
                'evaluation_success': False,
                'error': str(e)
            }
    
    return results

def generate_test_report(validation_results: dict, training_results: dict, 
                        evaluation_results: dict, output_path: str):
    """Generate comprehensive test report"""
    
    report = {
        'test_summary': {
            'total_formats': len(GoalConditioningManager.SUPPORTED_FORMATS),
            'validation_passed': sum(1 for r in validation_results.values() if r.get('compatible', False)),
            'training_passed': sum(1 for r in training_results.values() if r.get('training_success', False)),
            'evaluation_passed': sum(1 for r in evaluation_results.values() if r.get('evaluation_success', False))
        },
        'format_details': {},
        'recommendations': []
    }
    
    # Compile per-format results
    for format_type in GoalConditioningManager.SUPPORTED_FORMATS:
        format_report = {
            'validation': validation_results.get(format_type, {}),
            'training': training_results.get(format_type, {}),
            'evaluation': evaluation_results.get(format_type, {})
        }
        
        # Determine overall status
        validation_ok = format_report['validation'].get('compatible', False)
        training_ok = format_report['training'].get('training_success', True)  # Default True if not tested
        evaluation_ok = format_report['evaluation'].get('evaluation_success', True)  # Default True if not tested
        
        format_report['overall_status'] = 'PASS' if (validation_ok and training_ok and evaluation_ok) else 'FAIL'
        
        report['format_details'][format_type] = format_report
    
    # Generate recommendations
    failed_validation = [fmt for fmt, res in validation_results.items() if not res.get('compatible', False)]
    if failed_validation:
        report['recommendations'].append(f"Data incompatible with formats: {failed_validation}")
    
    failed_training = [fmt for fmt, res in training_results.items() if not res.get('training_success', True)]
    if failed_training:
        report['recommendations'].append(f"Training failed for formats: {failed_training}")
    
    failed_evaluation = [fmt for fmt, res in evaluation_results.items() if not res.get('evaluation_success', True)]
    if failed_evaluation:
        report['recommendations'].append(f"Evaluation failed for formats: {failed_evaluation}")
    
    if not report['recommendations']:
        report['recommendations'].append("All tested formats working correctly!")
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def print_test_summary(report: dict):
    """Print formatted test summary"""
    print(f"\n" + "="*80)
    print(f"🧪 COMPREHENSIVE TEST SUMMARY")
    print(f"="*80)
    
    summary = report['test_summary']
    print(f"Total Formats: {summary['total_formats']}")
    print(f"Validation Passed: {summary['validation_passed']}/{summary['total_formats']}")
    print(f"Training Passed: {summary['training_passed']}/{len([f for f in report['format_details'] if 'training' in report['format_details'][f]])}")
    print(f"Evaluation Passed: {summary['evaluation_passed']}/{len([f for f in report['format_details'] if 'evaluation' in report['format_details'][f]])}")
    
    print(f"\n📋 FORMAT STATUS:")
    print(f"-"*60)
    print(f"{'Format':<25} {'Validation':<12} {'Training':<10} {'Evaluation':<12} {'Status':<8}")
    print(f"-"*60)
    
    for format_type, details in report['format_details'].items():
        val_status = "✅" if details['validation'].get('compatible', False) else "❌"
        train_status = "✅" if details['training'].get('training_success', True) else "❌"
        eval_status = "✅" if details['evaluation'].get('evaluation_success', True) else "❌"
        overall = details['overall_status']
        
        # Show N/A for untested
        if not details['training']:
            train_status = "N/A"
        if not details['evaluation']:
            eval_status = "N/A"
        
        print(f"{format_type:<25} {val_status:<12} {train_status:<10} {eval_status:<12} {overall:<8}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    print(f"="*80)

def main():
    parser = argparse.ArgumentParser(description='Test all goal conditioning formats')
    parser.add_argument('--data_path', type=str, 
                      help='Path to test data file')
    parser.add_argument('--quick_test', action='store_true',
                      help='Only run validation tests (no training/evaluation)')
    parser.add_argument('--test_dir', type=str, default=None,
                      help='Directory for test outputs (default: temp dir)')
    parser.add_argument('--explain_formats', action='store_true',
                      help='Print explanation of all formats')
    
    args = parser.parse_args()
    
    if args.explain_formats:
        print_format_explanation()
        return
    
    if not args.data_path and not args.quick_test:
        print("❌ Error: --data_path required (or use --quick_test)")
        return
    
    # Create test directory
    if args.test_dir:
        test_dir = args.test_dir
        os.makedirs(test_dir, exist_ok=True)
    else:
        test_dir = tempfile.mkdtemp(prefix='bc_format_test_')
    
    print(f"🧪 COMPREHENSIVE GOAL CONDITIONING FORMAT TEST")
    print(f"Test directory: {test_dir}")
    
    # Run tests
    validation_results = {}
    training_results = {}
    evaluation_results = {}
    
    if args.data_path:
        # Test validation
        validation_results = test_format_validation(args.data_path)
        
        if not args.quick_test:
            # Test training (only if validation passed)
            compatible_formats = [fmt for fmt, res in validation_results.items() 
                                if res.get('compatible', False)]
            
            if compatible_formats:
                training_results = test_training_compatibility(args.data_path, test_dir)
                
                # Test evaluation
                if training_results:
                    evaluation_results = test_evaluation_compatibility(training_results, test_dir)
    
    # Generate and save report
    report_path = os.path.join(test_dir, 'format_test_report.json')
    report = generate_test_report(
        validation_results, training_results, evaluation_results, report_path
    )
    
    # Print summary
    print_test_summary(report)
    
    print(f"\n📁 Full test report saved to: {report_path}")
    print(f"🗂️  Test files in: {test_dir}")

if __name__ == '__main__':
    main() 