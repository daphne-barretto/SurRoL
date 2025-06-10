#!/usr/bin/env python3
"""
Corrected 100-Episode Analysis for CS224R Final Report
Analyzes spatial vs semantic conditioning with/without color information
Based on actual 100-episode evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

class Corrected100EpisodeAnalysis:
    def __init__(self):
        # Corrected 100-episode evaluation results
        self.evaluation_results = {
            'spatial_no_color': [10.0, 12.0, 8.0],      # targetblocktargetpeg runs 1-3
            'spatial_with_color': [6.0, 2.0, 5.0],      # targetblocktargetpeg_color runs 1-3
            'semantic_no_color': [3.0, 13.0, 7.0],      # onehottargetpeg runs 1-3  
            'semantic_with_color': [1.0, 6.0, 1.0]      # onehottargetpeg_color runs 1-3 (CORRECTED: 1/100 = 1.0%)
        }
        
        # Training results from previous analysis
        self.training_results = {
            'spatial_no_color': [35.0, 5.0, 15.0],      
            'spatial_with_color': [15.0, 10.0, 20.0],    
            'semantic_no_color': [20.0, 15.0, 20.0],     
            'semantic_with_color': [15.0, 40.0, 20.0]    
        }
        
        # Experiment name mapping
        self.experiment_mapping = {
            'spatial_no_color': [
                'bc_results_2block_targetblocktargetpeg_5k',
                'bc_results_2block_targetblocktargetpeg_5k_2', 
                'bc_results_2block_targetblocktargetpeg_5k_3'
            ],
            'spatial_with_color': [
                'bc_results_2block_targetblocktargetpeg_color_5k',
                'bc_results_2block_targetblocktargetpeg_color_5k_2',
                'bc_results_2block_targetblocktargetpeg_color_5k_3'
            ],
            'semantic_no_color': [
                'bc_results_2block_onehottargetpeg_5k',
                'bc_results_2block_onehottargetpeg_5k_2',
                'bc_results_2block_onehottargetpeg_5k_3'
            ],
            'semantic_with_color': [
                'bc_results_2block_onehottargetpeg_color_5k',
                'bc_results_2block_onehottargetpeg_color_5k_2', 
                'bc_results_2block_onehottargetpeg_color_5k_3'
            ]
        }
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis of 100-episode evaluation results"""
        print("🎯 CORRECTED 100-EPISODE EVALUATION ANALYSIS")
        print("=" * 80)
        
        # Calculate statistics for both training and evaluation
        eval_stats = self._calculate_statistics(self.evaluation_results, "Evaluation")
        train_stats = self._calculate_statistics(self.training_results, "Training")
        
        # Create visualizations
        self._create_comprehensive_plots(eval_stats, train_stats)
        
        # Generate insights
        insights = self._generate_insights(eval_stats, train_stats)
        
        # Create detailed tables
        self._create_detailed_tables(eval_stats, train_stats)
        
        # Print analysis
        self._print_comprehensive_analysis(eval_stats, train_stats, insights)
        
        return eval_stats, train_stats, insights
    
    def _calculate_statistics(self, results, data_type):
        """Calculate detailed statistics for each conditioning method"""
        stats = {}
        
        for method, results_list in results.items():
            stats[method] = {
                'method_name': self._get_method_name(method),
                'has_color': 'color' in method,
                'conditioning_type': 'Spatial' if 'spatial' in method else 'Semantic',
                'results': results_list,
                'mean': np.mean(results_list),
                'std': np.std(results_list),
                'min': np.min(results_list),
                'max': np.max(results_list),
                'median': np.median(results_list),
                'n_runs': len(results_list),
                'data_type': data_type
            }
        
        return stats
    
    def _get_method_name(self, method_key):
        """Get human-readable method name"""
        if 'spatial' in method_key:
            base = "Spatial (Target Block + Target Peg)"
        else:
            base = "Semantic (One Hot + Target Peg)"
        
        color_info = " with Color" if 'color' in method_key else " without Color"
        return base + color_info
    
    def _create_comprehensive_plots(self, eval_stats, train_stats):
        """Create comprehensive analysis plots"""
        print("\n📊 Creating comprehensive 100-episode analysis plots...")
        
        # Create main comparison figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Comprehensive Analysis: Training vs 100-Episode Evaluation', fontsize=16, fontweight='bold')
        
        # Plot 1: Training vs Evaluation Comparison
        ax1 = axes[0, 0]
        self._plot_training_vs_evaluation(ax1, eval_stats, train_stats)
        
        # Plot 2: Method Performance Comparison
        ax2 = axes[0, 1]
        self._plot_method_comparison(ax2, eval_stats)
        
        # Plot 3: Color Impact Analysis
        ax3 = axes[1, 0]
        self._plot_color_impact_analysis(ax3, eval_stats, train_stats)
        
        # Plot 4: Performance Distribution
        ax4 = axes[1, 1]
        self._plot_performance_distribution(ax4, eval_stats)
        
        # Plot 5: Individual Runs Comparison
        ax5 = axes[2, 0]
        self._plot_individual_runs(ax5, eval_stats, train_stats)
        
        # Plot 6: Consistency Analysis
        ax6 = axes[2, 1]
        self._plot_consistency_analysis(ax6, eval_stats, train_stats)
        
        plt.tight_layout()
        plt.savefig('comprehensive_100ep_corrected_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('comprehensive_100ep_corrected_analysis.pdf', bbox_inches='tight')
        print("✅ Comprehensive analysis plots saved!")
        
        # Create detailed heatmap
        self._create_detailed_heatmap(eval_stats, train_stats)
    
    def _plot_training_vs_evaluation(self, ax, eval_stats, train_stats):
        """Plot training vs evaluation performance"""
        methods = list(eval_stats.keys())
        train_means = [train_stats[method]['mean'] for method in methods]
        eval_means = [eval_stats[method]['mean'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        colors_train = ['#FF9999', '#FF6666', '#9999FF', '#6666FF']
        colors_eval = ['#66B2FF', '#4D9FFF', '#66FF66', '#4DFF4D']
        
        bars1 = ax.bar(x - width/2, train_means, width, label='Training Best', color=colors_train, alpha=0.8)
        bars2 = ax.bar(x + width/2, eval_means, width, label='100-Episode Eval', color=colors_eval, alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Conditioning Method')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Training vs 100-Episode Evaluation Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([eval_stats[method]['method_name'].replace(' (', '\n(') for method in methods], fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_method_comparison(self, ax, eval_stats):
        """Plot method comparison for evaluation results"""
        methods = [eval_stats[method]['method_name'].replace(' (', '\n(') for method in eval_stats.keys()]
        means = [eval_stats[method]['mean'] for method in eval_stats.keys()]
        stds = [eval_stats[method]['std'] for method in eval_stats.keys()]
        colors = ['#FF6B6B' if eval_stats[method]['has_color'] else '#4ECDC4' for method in eval_stats.keys()]
        
        bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.2,
                   f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('100-Episode Evaluation Performance')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_color_impact_analysis(self, ax, eval_stats, train_stats):
        """Plot color impact analysis for both training and evaluation"""
        # Calculate color impacts
        spatial_train_impact = train_stats['spatial_with_color']['mean'] - train_stats['spatial_no_color']['mean']
        spatial_eval_impact = eval_stats['spatial_with_color']['mean'] - eval_stats['spatial_no_color']['mean']
        semantic_train_impact = train_stats['semantic_with_color']['mean'] - train_stats['semantic_no_color']['mean']
        semantic_eval_impact = eval_stats['semantic_with_color']['mean'] - eval_stats['semantic_no_color']['mean']
        
        categories = ['Spatial', 'Semantic']
        train_impacts = [spatial_train_impact, semantic_train_impact]
        eval_impacts = [spatial_eval_impact, semantic_eval_impact]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_impacts, width, label='Training Impact', alpha=0.8, color='#FF9999')
        bars2 = ax.bar(x + width/2, eval_impacts, width, label='Evaluation Impact', alpha=0.8, color='#66B2FF')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 1,
                       f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax.set_xlabel('Conditioning Method')
        ax.set_ylabel('Color Impact (% Change)')
        ax.set_title('Color Information Impact: Training vs Evaluation')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    def _plot_performance_distribution(self, ax, eval_stats):
        """Plot performance distribution as violin plots"""
        data = []
        labels = []
        
        for method, data_stats in eval_stats.items():
            data.append(data_stats['results'])
            labels.append(data_stats['method_name'].replace(' (', '\n('))
        
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['#FF6B6B' if eval_stats[method]['has_color'] else '#4ECDC4' for method in eval_stats.keys()]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.8)
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('100-Episode Performance Distribution')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_individual_runs(self, ax, eval_stats, train_stats):
        """Plot individual run comparisons"""
        methods = list(eval_stats.keys())
        
        for i, method in enumerate(methods):
            train_runs = train_stats[method]['results']
            eval_runs = eval_stats[method]['results']
            
            x_pos = [i * 4 + j for j in range(3)]
            
            # Plot training runs
            ax.scatter(x_pos, train_runs, color='red', s=100, alpha=0.7, marker='o', label='Training' if i == 0 else "")
            # Plot evaluation runs
            ax.scatter(x_pos, eval_runs, color='blue', s=100, alpha=0.7, marker='s', label='Evaluation' if i == 0 else "")
            
            # Connect corresponding runs
            for j in range(3):
                ax.plot([x_pos[j], x_pos[j]], [train_runs[j], eval_runs[j]], 'gray', alpha=0.5, linestyle='--')
        
        # Set labels
        method_labels = [eval_stats[method]['method_name'].replace(' (', '\n(') for method in methods]
        ax.set_xticks([i * 4 + 1 for i in range(len(methods))])
        ax.set_xticklabels(method_labels, fontsize=8)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Individual Run Comparison: Training vs Evaluation')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_consistency_analysis(self, ax, eval_stats, train_stats):
        """Plot consistency analysis"""
        methods = [eval_stats[method]['method_name'].replace(' (', '\n(') for method in eval_stats.keys()]
        train_stds = [train_stats[method]['std'] for method in eval_stats.keys()]
        eval_stds = [eval_stats[method]['std'] for method in eval_stats.keys()]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_stds, width, label='Training Std', alpha=0.8, color='#FF9999')
        bars2 = ax.bar(x + width/2, eval_stds, width, label='Evaluation Std', alpha=0.8, color='#66B2FF')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Conditioning Method')
        ax.set_ylabel('Standard Deviation (%)')
        ax.set_title('Performance Consistency: Training vs Evaluation')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _create_detailed_heatmap(self, eval_stats, train_stats):
        """Create detailed heatmap comparing all results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Detailed Performance Heatmap: Training vs Evaluation', fontsize=14, fontweight='bold')
        
        methods = list(eval_stats.keys())
        method_names = [eval_stats[method]['method_name'] for method in methods]
        
        # Training heatmap
        train_matrix = np.array([train_stats[method]['results'] for method in methods])
        im1 = ax1.imshow(train_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=40)
        
        ax1.set_xticks(range(3))
        ax1.set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(method_names)
        ax1.set_title('Training Performance (%)')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(3):
                text = ax1.text(j, i, f'{train_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        # Evaluation heatmap
        eval_matrix = np.array([eval_stats[method]['results'] for method in methods])
        im2 = ax2.imshow(eval_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=15)
        
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(method_names)
        ax2.set_title('100-Episode Evaluation Performance (%)')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(3):
                text = ax2.text(j, i, f'{eval_matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Success Rate (%)')
        plt.colorbar(im2, ax=ax2, label='Success Rate (%)')
        
        plt.tight_layout()
        plt.savefig('detailed_performance_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Detailed performance heatmap saved!")
    
    def _generate_insights(self, eval_stats, train_stats):
        """Generate key insights from the analysis"""
        insights = []
        
        # Best performing methods
        best_eval = max(eval_stats.items(), key=lambda x: x[1]['mean'])
        best_train = max(train_stats.items(), key=lambda x: x[1]['mean'])
        
        insights.append(f"🏆 Best Evaluation Method: {best_eval[1]['method_name']} ({best_eval[1]['mean']:.1f}%)")
        insights.append(f"🏆 Best Training Method: {best_train[1]['method_name']} ({best_train[1]['mean']:.1f}%)")
        
        # Spatial vs Semantic comparison
        spatial_eval_avg = np.mean([eval_stats['spatial_no_color']['mean'], eval_stats['spatial_with_color']['mean']])
        semantic_eval_avg = np.mean([eval_stats['semantic_no_color']['mean'], eval_stats['semantic_with_color']['mean']])
        
        if spatial_eval_avg > semantic_eval_avg:
            improvement = ((spatial_eval_avg - semantic_eval_avg) / semantic_eval_avg * 100)
            insights.append(f"🎯 Spatial outperforms Semantic in evaluation by {improvement:.1f}% ({spatial_eval_avg:.1f}% vs {semantic_eval_avg:.1f}%)")
        else:
            improvement = ((semantic_eval_avg - spatial_eval_avg) / spatial_eval_avg * 100)
            insights.append(f"🎯 Semantic outperforms Spatial in evaluation by {improvement:.1f}% ({semantic_eval_avg:.1f}% vs {spatial_eval_avg:.1f}%)")
        
        # Color impact
        spatial_color_impact = eval_stats['spatial_with_color']['mean'] - eval_stats['spatial_no_color']['mean']
        semantic_color_impact = eval_stats['semantic_with_color']['mean'] - eval_stats['semantic_no_color']['mean']
        
        insights.append(f"🌈 Color impact on spatial (evaluation): {spatial_color_impact:+.1f}%")
        insights.append(f"🌈 Color impact on semantic (evaluation): {semantic_color_impact:+.1f}%")
        
        # Training vs Evaluation gaps
        for method in eval_stats.keys():
            gap = train_stats[method]['mean'] - eval_stats[method]['mean']
            insights.append(f"📊 {eval_stats[method]['method_name']} gap (Training-Eval): {gap:+.1f}%")
        
        return insights
    
    def _create_detailed_tables(self, eval_stats, train_stats):
        """Create comprehensive tables"""
        print("\n📋 Creating detailed comparison tables...")
        
        # Main comparison table
        comparison_data = []
        for method in eval_stats.keys():
            comparison_data.append({
                'Method': eval_stats[method]['method_name'],
                'Conditioning': eval_stats[method]['conditioning_type'],
                'Color': 'Yes' if eval_stats[method]['has_color'] else 'No',
                'Training Mean (%)': f"{train_stats[method]['mean']:.1f}",
                'Training Std (%)': f"{train_stats[method]['std']:.1f}",
                'Training Results (%)': ', '.join([f"{r:.1f}" for r in train_stats[method]['results']]),
                'Evaluation Mean (%)': f"{eval_stats[method]['mean']:.1f}",
                'Evaluation Std (%)': f"{eval_stats[method]['std']:.1f}",
                'Evaluation Results (%)': ', '.join([f"{r:.1f}" for r in eval_stats[method]['results']]),
                'Performance Gap (%)': f"{(train_stats[method]['mean'] - eval_stats[method]['mean']):+.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('comprehensive_training_vs_evaluation_comparison.csv', index=False)
        
        # Summary statistics table
        summary_data = []
        for data_type, stats in [('Training', train_stats), ('Evaluation', eval_stats)]:
            for method, data in stats.items():
                summary_data.append({
                    'Data Type': data_type,
                    'Method': data['method_name'],
                    'Mean (%)': f"{data['mean']:.1f}",
                    'Std (%)': f"{data['std']:.1f}",
                    'Min (%)': f"{data['min']:.1f}",
                    'Max (%)': f"{data['max']:.1f}",
                    'Range (%)': f"{data['max'] - data['min']:.1f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('detailed_summary_statistics.csv', index=False)
        
        print("✅ Detailed tables saved!")
        return comparison_df, summary_df
    
    def _print_comprehensive_analysis(self, eval_stats, train_stats, insights):
        """Print comprehensive analysis results"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TRAINING VS 100-EPISODE EVALUATION ANALYSIS")
        print("="*80)
        
        print(f"\n📊 EVALUATION RESULTS (100 episodes each):")
        for method, data in eval_stats.items():
            print(f"\n  {data['method_name']}:")
            print(f"    • Evaluation: {data['mean']:.1f}% ± {data['std']:.1f}%")
            print(f"    • Training: {train_stats[method]['mean']:.1f}% ± {train_stats[method]['std']:.1f}%")
            print(f"    • Gap: {(train_stats[method]['mean'] - data['mean']):+.1f}%")
            print(f"    • Evaluation runs: {[f'{r:.1f}%' for r in data['results']]}")
        
        print(f"\n🔬 KEY INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
        
        print(f"\n📈 PERFORMANCE RANKING (Evaluation):")
        ranked_methods = sorted(eval_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, data) in enumerate(ranked_methods, 1):
            print(f"  {i}. {data['method_name']}: {data['mean']:.1f}%")
        
        print("="*80)

def main():
    """Main analysis function"""
    print("🔬 CORRECTED 100-EPISODE EVALUATION ANALYSIS")
    print("=" * 80)
    
    analyzer = Corrected100EpisodeAnalysis()
    
    # Run comprehensive analysis
    eval_stats, train_stats, insights = analyzer.create_comprehensive_analysis()
    
    print(f"\n📊 ANALYSIS COMPLETE!")
    print(f"Files generated:")
    print(f"  • comprehensive_100ep_corrected_analysis.png/pdf - Main analysis")
    print(f"  • detailed_performance_heatmap.png - Performance heatmap")
    print(f"  • comprehensive_training_vs_evaluation_comparison.csv - Detailed comparison")
    print(f"  • detailed_summary_statistics.csv - Summary statistics")
    
    return eval_stats, train_stats, insights

if __name__ == "__main__":
    eval_stats, train_stats, insights = main() 