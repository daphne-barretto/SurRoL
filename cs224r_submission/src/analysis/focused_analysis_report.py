#!/usr/bin/env python3
"""
Focused Analysis for CS224R Final Report
Analyzes spatial vs semantic conditioning with/without color information
Based on recovered training data from 12 experiments
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

class FocusedAnalysisReport:
    def __init__(self):
        self.experiments_dir = Path("behavior-cloning/hw1/cs224r/experiments/5k_demos")
        
        # Recovered training performance data
        self.training_results = {
            'spatial_no_color': [0.35, 0.05, 0.15],      # runs 1-3
            'spatial_with_color': [0.15, 0.10, 0.20],    # runs 1-3
            'semantic_no_color': [0.20, 0.15, 0.20],     # runs 1-3  
            'semantic_with_color': [0.15, 0.40, 0.20]    # runs 1-3
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
    
    def create_comprehensive_training_analysis(self):
        """Create comprehensive analysis of training performance"""
        print("🎯 FOCUSED ANALYSIS: SPATIAL VS SEMANTIC CONDITIONING")
        print("=" * 80)
        
        # Calculate statistics
        stats = self._calculate_group_statistics()
        
        # Create visualizations
        self._create_training_performance_plots(stats)
        
        # Generate insights
        insights = self._generate_key_insights(stats)
        
        # Create summary tables
        self._create_summary_tables(stats)
        
        # Print detailed analysis
        self._print_detailed_analysis(stats, insights)
        
        return stats, insights
    
    def _calculate_group_statistics(self):
        """Calculate detailed statistics for each conditioning method"""
        stats = {}
        
        for method, results in self.training_results.items():
            stats[method] = {
                'method_name': self._get_method_name(method),
                'has_color': 'color' in method,
                'results': results,
                'mean': np.mean(results),
                'std': np.std(results),
                'min': np.min(results),
                'max': np.max(results),
                'median': np.median(results),
                'n_runs': len(results)
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
    
    def _create_training_performance_plots(self, stats):
        """Create comprehensive training performance plots"""
        print("\n📊 Creating training performance visualizations...")
        
        # Set up the plotting style - fix compatibility issue
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
        
        # Create main comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Performance Analysis: Spatial vs Semantic Conditioning', fontsize=16, fontweight='bold')
        
        # Plot 1: Mean Performance Comparison
        ax1 = axes[0, 0]
        self._plot_mean_performance(ax1, stats)
        
        # Plot 2: Performance Distribution
        ax2 = axes[0, 1]
        self._plot_performance_distribution(ax2, stats)
        
        # Plot 3: Color Impact Analysis
        ax3 = axes[0, 2]
        self._plot_color_impact(ax3, stats)
        
        # Plot 4: Individual Run Results
        ax4 = axes[1, 0]
        self._plot_individual_runs(ax4, stats)
        
        # Plot 5: Variance Analysis
        ax5 = axes[1, 1]
        self._plot_variance_analysis(ax5, stats)
        
        # Plot 6: Method Comparison Matrix
        ax6 = axes[1, 2]
        self._plot_method_comparison_matrix(ax6, stats)
        
        plt.tight_layout()
        plt.savefig('focused_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('focused_training_analysis.pdf', bbox_inches='tight')
        print("✅ Training analysis plots saved!")
        
        # Create detailed performance breakdown
        self._create_detailed_performance_breakdown(stats)
    
    def _plot_mean_performance(self, ax, stats):
        """Plot mean performance comparison"""
        methods = []
        means = []
        stds = []
        colors = []
        
        for method, data in stats.items():
            methods.append(data['method_name'].replace(' (', '\n('))
            means.append(data['mean'])
            stds.append(data['std'])
            colors.append('#FF6B6B' if data['has_color'] else '#4ECDC4')
        
        bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Success Rate')
        ax.set_title('Mean Training Performance')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(means) + max(stds) + 0.05)
    
    def _plot_performance_distribution(self, ax, stats):
        """Plot performance distribution as box plots"""
        data = []
        labels = []
        
        for method, data_stats in stats.items():
            data.append(data_stats['results'])
            labels.append(data_stats['method_name'].replace(' (', '\n('))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['#FF6B6B' if stats[method]['has_color'] else '#4ECDC4' 
                 for method in stats.keys()]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_ylabel('Success Rate')
        ax.set_title('Performance Distribution')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_color_impact(self, ax, stats):
        """Plot color information impact"""
        spatial_no_color = stats['spatial_no_color']['mean']
        spatial_with_color = stats['spatial_with_color']['mean']
        semantic_no_color = stats['semantic_no_color']['mean']  
        semantic_with_color = stats['semantic_with_color']['mean']
        
        methods = ['Spatial', 'Semantic']
        no_color_vals = [spatial_no_color, semantic_no_color]
        with_color_vals = [spatial_with_color, semantic_with_color]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, no_color_vals, width, label='No Color', color='#4ECDC4', alpha=0.8)
        bars2 = ax.bar(x + width/2, with_color_vals, width, label='With Color', color='#FF6B6B', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Conditioning Method')
        ax.set_ylabel('Success Rate')
        ax.set_title('Color Information Impact')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_individual_runs(self, ax, stats):
        """Plot individual run results"""
        x_offset = 0
        colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#FFA07A']
        
        for i, (method, data) in enumerate(stats.items()):
            runs = data['results']
            x_pos = [x_offset + j for j in range(len(runs))]
            
            ax.scatter(x_pos, runs, label=data['method_name'], 
                      color=colors[i], s=80, alpha=0.8, edgecolor='black')
            
            # Add run labels
            for j, (x, y) in enumerate(zip(x_pos, runs)):
                ax.annotate(f'R{j+1}', (x, y), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=8)
            
            x_offset += len(runs) + 1
        
        ax.set_xlabel('Individual Runs')
        ax.set_ylabel('Success Rate')
        ax.set_title('Individual Run Performance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
    
    def _plot_variance_analysis(self, ax, stats):
        """Plot performance variance analysis"""
        methods = [stats[method]['method_name'].replace(' (', '\n(') for method in stats.keys()]
        variances = [stats[method]['std'] for method in stats.keys()]
        colors = ['#FF6B6B' if stats[method]['has_color'] else '#4ECDC4' for method in stats.keys()]
        
        bars = ax.bar(methods, variances, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{var:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Performance Consistency')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_method_comparison_matrix(self, ax, stats):
        """Plot method comparison heatmap"""
        methods = list(stats.keys())
        n_methods = len(methods)
        
        # Create performance matrix
        performance_matrix = np.zeros((n_methods, 3))  # 3 runs each
        
        for i, method in enumerate(methods):
            performance_matrix[i] = stats[method]['results']
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.4)
        
        # Set labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(['Run 1', 'Run 2', 'Run 3'])
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels([stats[method]['method_name'].replace(' (', '\n(') for method in methods])
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(3):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.1%}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Performance Heatmap')
        plt.colorbar(im, ax=ax, label='Success Rate')
    
    def _create_detailed_performance_breakdown(self, stats):
        """Create detailed performance breakdown visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        categories = ['Spatial\nNo Color', 'Spatial\nWith Color', 'Semantic\nNo Color', 'Semantic\nWith Color']
        run1_data = [stats[method]['results'][0] for method in stats.keys()]
        run2_data = [stats[method]['results'][1] for method in stats.keys()]  
        run3_data = [stats[method]['results'][2] for method in stats.keys()]
        
        x = np.arange(len(categories))
        width = 0.25
        
        bars1 = ax.bar(x - width, run1_data, width, label='Run 1', alpha=0.8)
        bars2 = ax.bar(x, run2_data, width, label='Run 2', alpha=0.8)
        bars3 = ax.bar(x + width, run3_data, width, label='Run 3', alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Conditioning Method')
        ax.set_ylabel('Training Success Rate')
        ax.set_title('Detailed Performance Breakdown by Run')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_performance_breakdown.png', dpi=300, bbox_inches='tight')
        print("✅ Detailed performance breakdown saved!")
    
    def _generate_key_insights(self, stats):
        """Generate key insights from the analysis"""
        insights = []
        
        # Overall best method
        best_method = max(stats.items(), key=lambda x: x[1]['mean'])
        insights.append(f"🏆 Best Overall Method: {best_method[1]['method_name']} ({best_method[1]['mean']:.1%} mean)")
        
        # Spatial vs Semantic comparison
        spatial_avg = np.mean([stats['spatial_no_color']['mean'], stats['spatial_with_color']['mean']])
        semantic_avg = np.mean([stats['semantic_no_color']['mean'], stats['semantic_with_color']['mean']])
        
        if spatial_avg > semantic_avg:
            improvement = ((spatial_avg - semantic_avg) / semantic_avg * 100)
            insights.append(f"🎯 Spatial conditioning outperforms semantic by {improvement:.1f}% ({spatial_avg:.1%} vs {semantic_avg:.1%})")
        else:
            improvement = ((semantic_avg - spatial_avg) / spatial_avg * 100)
            insights.append(f"🎯 Semantic conditioning outperforms spatial by {improvement:.1f}% ({semantic_avg:.1%} vs {spatial_avg:.1%})")
        
        # Color impact analysis
        spatial_color_impact = stats['spatial_with_color']['mean'] - stats['spatial_no_color']['mean']
        semantic_color_impact = stats['semantic_with_color']['mean'] - stats['semantic_no_color']['mean']
        
        insights.append(f"🌈 Color impact on spatial conditioning: {spatial_color_impact:+.1%}")
        insights.append(f"🌈 Color impact on semantic conditioning: {semantic_color_impact:+.1%}")
        
        # Consistency analysis
        most_consistent = min(stats.items(), key=lambda x: x[1]['std'])
        least_consistent = max(stats.items(), key=lambda x: x[1]['std'])
        
        insights.append(f"📊 Most consistent: {most_consistent[1]['method_name']} (std: {most_consistent[1]['std']:.3f})")
        insights.append(f"📊 Least consistent: {least_consistent[1]['method_name']} (std: {least_consistent[1]['std']:.3f})")
        
        # Performance ranges
        for method, data in stats.items():
            range_val = data['max'] - data['min']
            insights.append(f"📈 {data['method_name']}: Range {data['min']:.1%}-{data['max']:.1%} (span: {range_val:.1%})")
        
        return insights
    
    def _create_summary_tables(self, stats):
        """Create comprehensive summary tables"""
        print("\n📋 Creating summary tables...")
        
        # Main summary table
        summary_data = []
        for method, data in stats.items():
            summary_data.append({
                'Method': data['method_name'],
                'Color': 'Yes' if data['has_color'] else 'No',
                'Mean (%)': f"{data['mean']*100:.1f}",
                'Std (%)': f"{data['std']*100:.1f}",
                'Min (%)': f"{data['min']*100:.1f}",
                'Max (%)': f"{data['max']*100:.1f}",
                'Median (%)': f"{data['median']*100:.1f}",
                'Runs': data['n_runs'],
                'Individual Results (%)': ', '.join([f"{r*100:.1f}" for r in data['results']])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('training_performance_summary.csv', index=False)
        
        # Comparison table
        comparison_data = [
            {
                'Comparison': 'Spatial vs Semantic (No Color)',
                'Spatial': f"{stats['spatial_no_color']['mean']:.1%}",
                'Semantic': f"{stats['semantic_no_color']['mean']:.1%}",
                'Difference': f"{(stats['spatial_no_color']['mean'] - stats['semantic_no_color']['mean']):+.1%}"
            },
            {
                'Comparison': 'Spatial vs Semantic (With Color)',
                'Spatial': f"{stats['spatial_with_color']['mean']:.1%}",
                'Semantic': f"{stats['semantic_with_color']['mean']:.1%}",
                'Difference': f"{(stats['spatial_with_color']['mean'] - stats['semantic_with_color']['mean']):+.1%}"
            },
            {
                'Comparison': 'Color Impact (Spatial)',
                'No Color': f"{stats['spatial_no_color']['mean']:.1%}",
                'With Color': f"{stats['spatial_with_color']['mean']:.1%}",
                'Difference': f"{(stats['spatial_with_color']['mean'] - stats['spatial_no_color']['mean']):+.1%}"
            },
            {
                'Comparison': 'Color Impact (Semantic)',
                'No Color': f"{stats['semantic_no_color']['mean']:.1%}",
                'With Color': f"{stats['semantic_with_color']['mean']:.1%}",
                'Difference': f"{(stats['semantic_with_color']['mean'] - stats['semantic_no_color']['mean']):+.1%}"
            }
        ]
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('method_comparison_table.csv', index=False)
        
        print("✅ Summary tables saved!")
        return summary_df, comparison_df
    
    def _print_detailed_analysis(self, stats, insights):
        """Print comprehensive analysis results"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TRAINING PERFORMANCE ANALYSIS")
        print("="*80)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        for method, data in stats.items():
            print(f"\n  {data['method_name']}:")
            print(f"    • Mean: {data['mean']:.1%} ± {data['std']:.1%}")
            print(f"    • Range: {data['min']:.1%} - {data['max']:.1%}")
            print(f"    • Individual runs: {[f'{r:.1%}' for r in data['results']]}")
        
        print(f"\n🔬 KEY INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
        
        print(f"\n📈 PERFORMANCE RANKING:")
        ranked_methods = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, data) in enumerate(ranked_methods, 1):
            print(f"  {i}. {data['method_name']}: {data['mean']:.1%}")
        
        print("="*80)
    
    def generate_cs224r_report_section(self, stats, insights):
        """Generate formatted section for CS224R final report"""
        print("\n📝 Generating CS224R Report Section...")
        
        report_content = f"""
# Goal Conditioning Analysis: Spatial vs Semantic Representations

## Methodology
We compared four goal conditioning approaches for surgical robotics manipulation:
1. **Spatial (Target Block + Target Peg) without color**: Direct spatial coordinates
2. **Spatial (Target Block + Target Peg) with color**: Spatial coordinates + RGBA encoding  
3. **Semantic (One Hot + Target Peg) without color**: Categorical representation + spatial target
4. **Semantic (One Hot + Target Peg) with color**: Categorical + spatial + RGBA encoding

Each method was trained for 3 independent runs using 5,000 demonstration episodes.

## Results

### Training Performance Summary
| Method | Color | Mean Success Rate | Std Dev | Individual Runs |
|--------|-------|------------------|---------|-----------------|
| Spatial | No | {stats['spatial_no_color']['mean']:.1%} | {stats['spatial_no_color']['std']:.3f} | {', '.join([f'{r:.1%}' for r in stats['spatial_no_color']['results']])} |
| Spatial | Yes | {stats['spatial_with_color']['mean']:.1%} | {stats['spatial_with_color']['std']:.3f} | {', '.join([f'{r:.1%}' for r in stats['spatial_with_color']['results']])} |
| Semantic | No | {stats['semantic_no_color']['mean']:.1%} | {stats['semantic_no_color']['std']:.3f} | {', '.join([f'{r:.1%}' for r in stats['semantic_no_color']['results']])} |
| Semantic | Yes | {stats['semantic_with_color']['mean']:.1%} | {stats['semantic_with_color']['std']:.3f} | {', '.join([f'{r:.1%}' for r in stats['semantic_with_color']['results']])} |

### Key Findings
"""
        
        for insight in insights:
            # Convert emoji insights to bullet points
            clean_insight = insight.replace('🏆', '•').replace('🎯', '•').replace('🌈', '•').replace('📊', '•').replace('📈', '•')
            report_content += f"{clean_insight}\n"
        
        # Calculate important metrics
        spatial_avg = np.mean([stats['spatial_no_color']['mean'], stats['spatial_with_color']['mean']])
        semantic_avg = np.mean([stats['semantic_no_color']['mean'], stats['semantic_with_color']['mean']])
        
        report_content += f"""
### Discussion

**Spatial vs Semantic Conditioning**: Our analysis reveals that {'spatial' if spatial_avg > semantic_avg else 'semantic'} conditioning achieves superior performance ({spatial_avg:.1%} vs {semantic_avg:.1%} mean success rate). This suggests that {'direct coordinate-based representations are more effective for this manipulation task' if spatial_avg > semantic_avg else 'categorical representations with spatial targets provide better generalization'}.

**Color Information Impact**: Color encoding shows differential effects across conditioning methods:
- Spatial conditioning: {(stats['spatial_with_color']['mean'] - stats['spatial_no_color']['mean']):+.1%} impact
- Semantic conditioning: {(stats['semantic_with_color']['mean'] - stats['semantic_no_color']['mean']):+.1%} impact

**Consistency Analysis**: Performance variance ranges from {min([data['std'] for data in stats.values()]):.3f} to {max([data['std'] for data in stats.values()]):.3f}, indicating {'high' if max([data['std'] for data in stats.values()]) > 0.1 else 'moderate'} variability across methods.

### Implications for Surgical Robotics
The results have important implications for goal-conditioned learning in surgical manipulation tasks, particularly for peg transfer exercises in robotic surgery training systems.
"""
        
        # Save report section
        with open('cs224r_report_section.md', 'w') as f:
            f.write(report_content)
        
        print("✅ CS224R Report section saved!")
        return report_content

def main():
    """Main analysis function"""
    print("🔬 FOCUSED ANALYSIS FOR CS224R FINAL REPORT")
    print("=" * 80)
    
    analyzer = FocusedAnalysisReport()
    
    # Run comprehensive analysis
    stats, insights = analyzer.create_comprehensive_training_analysis()
    
    # Generate report section
    report_content = analyzer.generate_cs224r_report_section(stats, insights)
    
    print(f"\n📊 ANALYSIS COMPLETE!")
    print(f"Files generated:")
    print(f"  • focused_training_analysis.png/pdf - Main analysis plots")
    print(f"  • detailed_performance_breakdown.png - Detailed breakdown")
    print(f"  • training_performance_summary.csv - Summary statistics")
    print(f"  • method_comparison_table.csv - Method comparisons")
    print(f"  • cs224r_report_section.md - Report section")
    
    return stats, insights

if __name__ == "__main__":
    stats, insights = main() 