#!/usr/bin/env python3
"""
Generate a clean visual table of corrected 100-episode evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_results_table():
    """Create comprehensive results table with corrected data"""
    
    # Corrected data extracted from the evaluation output
    data = {
        'Method': [
            'Spatial without Color',
            'Spatial with Color', 
            'Semantic without Color',
            'Semantic with Color'
        ],
        'Experiment Names': [
            'targetblocktargetpeg',
            'targetblocktargetpeg_color',
            'onehottargetpeg',
            'onehottargetpeg_color'
        ],
        'Training Mean (%)': [18.3, 15.0, 18.3, 25.0],
        'Training Std (%)': [12.5, 4.1, 2.4, 10.8],
        'Training Run 1 (%)': [35.0, 15.0, 20.0, 15.0],
        'Training Run 2 (%)': [5.0, 10.0, 15.0, 40.0],
        'Training Run 3 (%)': [15.0, 20.0, 20.0, 20.0],
        'Evaluation Mean (%)': [10.0, 4.3, 7.7, 2.7],
        'Evaluation Std (%)': [1.6, 1.7, 4.1, 2.4],
        'Evaluation Run 1 (%)': [10.0, 6.0, 3.0, 1.0],  # CORRECTED: 1/100 = 1.0%
        'Evaluation Run 2 (%)': [12.0, 2.0, 13.0, 6.0],
        'Evaluation Run 3 (%)': [8.0, 5.0, 7.0, 1.0],   # CORRECTED: 1/100 = 1.0%
        'Performance Gap (%)': [8.3, 10.7, 10.7, 22.3]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('corrected_comprehensive_results.csv', index=False)
    
    # Create visualization
    create_comparison_plots(df)
    
    return df

def create_comparison_plots(df):
    """Create visual comparison plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Corrected 100-Episode Evaluation Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training vs Evaluation Means
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Training Mean (%)'], width, label='Training', alpha=0.8, color='red')
    bars2 = ax1.bar(x + width/2, df['Evaluation Mean (%)'], width, label='Evaluation', alpha=0.8, color='blue')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Training vs Evaluation Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Performance Gaps
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Method'], df['Performance Gap (%)'], color=['green', 'orange', 'orange', 'red'], alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Performance Gap (%)')
    ax2.set_title('Training-Evaluation Performance Gap')
    ax2.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Individual Runs Comparison
    ax3 = axes[1, 0]
    
    # Training runs
    train_runs = [
        [35.0, 5.0, 15.0],   # Spatial no color
        [15.0, 10.0, 20.0],  # Spatial with color
        [20.0, 15.0, 20.0],  # Semantic no color
        [15.0, 40.0, 20.0]   # Semantic with color
    ]
    
    # Evaluation runs
    eval_runs = [
        [10.0, 12.0, 8.0],   # Spatial no color
        [6.0, 2.0, 5.0],     # Spatial with color
        [3.0, 13.0, 7.0],    # Semantic no color
        [1.0, 6.0, 1.0]      # Semantic with color - CORRECTED
    ]
    
    for i, method in enumerate(df['Method']):
        x_pos = [i * 4 + j for j in range(3)]
        ax3.scatter(x_pos, train_runs[i], color='red', s=100, alpha=0.7, marker='o', 
                   label='Training' if i == 0 else "")
        ax3.scatter(x_pos, eval_runs[i], color='blue', s=100, alpha=0.7, marker='s', 
                   label='Evaluation' if i == 0 else "")
        
        # Connect runs with lines
        for j in range(3):
            ax3.plot([x_pos[j], x_pos[j]], [train_runs[i][j], eval_runs[i][j]], 
                    'gray', alpha=0.5, linestyle='--')
    
    ax3.set_xticks([i * 4 + 1 for i in range(len(df))])
    ax3.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Individual Runs: Training vs Evaluation')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Ranking Comparison
    ax4 = axes[1, 1]
    
    # Create ranking comparison
    training_ranking = df.sort_values('Training Mean (%)', ascending=False)['Method'].values
    evaluation_ranking = df.sort_values('Evaluation Mean (%)', ascending=False)['Method'].values
    
    train_ranks = {method: i+1 for i, method in enumerate(training_ranking)}
    eval_ranks = {method: i+1 for i, method in enumerate(evaluation_ranking)}
    
    methods = df['Method']
    train_rank_values = [train_ranks[method] for method in methods]
    eval_rank_values = [eval_ranks[method] for method in methods]
    
    x_pos = np.arange(len(methods))
    ax4.scatter(x_pos, train_rank_values, color='red', s=150, alpha=0.7, marker='o', label='Training Rank')
    ax4.scatter(x_pos, eval_rank_values, color='blue', s=150, alpha=0.7, marker='s', label='Evaluation Rank')
    
    # Connect ranks with lines
    for i in range(len(methods)):
        ax4.plot([x_pos[i], x_pos[i]], [train_rank_values[i], eval_rank_values[i]], 
                'gray', alpha=0.5, linestyle='--')
        
        # Add rank labels
        ax4.text(x_pos[i] - 0.1, train_rank_values[i], f'{train_rank_values[i]}', 
                ha='center', va='center', fontweight='bold', color='red')
        ax4.text(x_pos[i] + 0.1, eval_rank_values[i], f'{eval_rank_values[i]}', 
                ha='center', va='center', fontweight='bold', color='blue')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Ranking (1=Best)')
    ax4.set_title('Performance Ranking: Training vs Evaluation')
    ax4.set_ylim(0.5, 4.5)
    ax4.invert_yaxis()  # Best rank (1) at top
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('corrected_comprehensive_comparison.pdf', bbox_inches='tight')
    print("✅ Corrected comprehensive comparison plots saved!")

def print_summary_table(df):
    """Print formatted summary table"""
    print("\n" + "="*120)
    print("🎯 CORRECTED 100-EPISODE EVALUATION RESULTS SUMMARY")
    print("="*120)
    
    print(f"\n{'Method':<25} {'Train Mean':<12} {'Eval Mean':<12} {'Gap':<8} {'Train Runs':<20} {'Eval Runs':<20}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        train_runs = f"{row['Training Run 1 (%)']:.1f}, {row['Training Run 2 (%)']:.1f}, {row['Training Run 3 (%)']:.1f}"
        eval_runs = f"{row['Evaluation Run 1 (%)']:.1f}, {row['Evaluation Run 2 (%)']:.1f}, {row['Evaluation Run 3 (%)']:.1f}"
        
        print(f"{row['Method']:<25} {row['Training Mean (%)']:>8.1f}% {row['Evaluation Mean (%)']:>8.1f}% "
              f"{row['Performance Gap (%)']:>6.1f}% {train_runs:<20} {eval_runs:<20}")
    
    print("\n" + "="*120)
    print("🔬 KEY FINDINGS:")
    print("  🏆 Best Evaluation: Spatial without Color (10.0%)")
    print("  📊 Spatial > Semantic: 7.2% vs 5.2% (38.7% advantage)")
    print("  🌈 Color Impact: Negative for both spatial (-5.7%) and semantic (-5.0%)")
    print("  📉 Worst Generalization: Semantic with Color (22.3% gap)")
    print("  ⚠️  Training ≠ Evaluation: Best trainer (25.0%) is worst evaluator (2.7%)")
    print("="*120)

def main():
    """Main function"""
    print("🔬 CREATING CORRECTED RESULTS TABLE AND ANALYSIS")
    
    df = create_results_table()
    print_summary_table(df)
    
    print(f"\n📊 FILES GENERATED:")
    print(f"  • corrected_comprehensive_results.csv - Raw data table")
    print(f"  • corrected_comprehensive_comparison.png/pdf - Visual analysis")
    
    return df

if __name__ == "__main__":
    df = main() 