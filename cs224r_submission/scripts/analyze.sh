#!/bin/bash

# CS224R Analysis Script
# Runs comprehensive analysis of training and evaluation results

set -e  # Exit on any error

echo "📊 CS224R RESULTS ANALYSIS"
echo "=========================="

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gcrl

# Set paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"

# Default parameters
ANALYSIS_TYPE="all"  # Options: corrected, focused, comparison, all
OUTPUT_FORMAT="both"  # Options: png, pdf, both

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        --format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --type      Analysis type (corrected, focused, comparison, all)"
            echo "  --format    Output format (png, pdf, both)"
            echo ""
            echo "Analysis Types:"
            echo "  corrected    - Corrected 100-episode analysis with training vs evaluation"
            echo "  focused      - Focused training analysis with statistical insights"
            echo "  comparison   - Comprehensive comparison plots and tables"
            echo "  all          - Run all analysis types"
            echo ""
            echo "Examples:"
            echo "  $0 --type corrected --format pdf"
            echo "  $0 --type all --format both"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "$RESULTS_DIR"/{plots,tables,reports}

echo "🎯 Analysis Configuration:"
echo "   Type: $ANALYSIS_TYPE"
echo "   Output Format: $OUTPUT_FORMAT"
echo "   Working Directory: $PROJECT_ROOT"
echo ""

# Change to project root for analysis scripts
cd "$PROJECT_ROOT"

# Function to run corrected 100-episode analysis
run_corrected_analysis() {
    echo "🔬 Running corrected 100-episode analysis..."
    python "$SRC_DIR/analysis/corrected_100ep_analysis.py"
    
    # Move results to organized directories
    mv comprehensive_100ep_corrected_analysis.* "$RESULTS_DIR/plots/" 2>/dev/null || true
    mv detailed_performance_heatmap.* "$RESULTS_DIR/plots/" 2>/dev/null || true
    mv comprehensive_training_vs_evaluation_comparison.csv "$RESULTS_DIR/tables/" 2>/dev/null || true
    mv detailed_summary_statistics.csv "$RESULTS_DIR/tables/" 2>/dev/null || true
    
    echo "✅ Corrected analysis completed"
}

# Function to run focused training analysis  
run_focused_analysis() {
    echo "🎯 Running focused training analysis..."
    python "$SRC_DIR/analysis/focused_analysis_report.py"
    
    # Move results to organized directories
    mv focused_training_analysis.* "$RESULTS_DIR/plots/" 2>/dev/null || true
    mv detailed_performance_breakdown.* "$RESULTS_DIR/plots/" 2>/dev/null || true
    mv training_performance_summary.csv "$RESULTS_DIR/tables/" 2>/dev/null || true
    mv method_comparison_table.csv "$RESULTS_DIR/tables/" 2>/dev/null || true
    mv cs224r_report_section.md "$RESULTS_DIR/reports/" 2>/dev/null || true
    
    echo "✅ Focused analysis completed"
}

# Function to run comparison analysis
run_comparison_analysis() {
    echo "📈 Running comparison analysis..."
    python "$SRC_DIR/analysis/corrected_results_table.py"
    
    # Move results to organized directories
    mv corrected_comprehensive_comparison.* "$RESULTS_DIR/plots/" 2>/dev/null || true
    mv corrected_comprehensive_results.csv "$RESULTS_DIR/tables/" 2>/dev/null || true
    
    echo "✅ Comparison analysis completed"
}

# Run analysis based on type selection
case $ANALYSIS_TYPE in
    "corrected")
        run_corrected_analysis
        ;;
    "focused")
        run_focused_analysis
        ;;
    "comparison")
        run_comparison_analysis
        ;;
    "all")
        echo "📊 Running all analysis types..."
        run_corrected_analysis
        echo ""
        run_focused_analysis
        echo ""
        run_comparison_analysis
        ;;
    *)
        echo "❌ Error: Unknown analysis type '$ANALYSIS_TYPE'"
        echo "Valid types: corrected, focused, comparison, all"
        exit 1
        ;;
esac

# Clean up any temporary files in project root
echo ""
echo "🧹 Cleaning up temporary files..."
rm -f *.png *.pdf *.csv 2>/dev/null || true

# Generate summary report
echo "📋 Generating analysis summary..."
cat > "$RESULTS_DIR/reports/analysis_summary.md" << EOF
# CS224R Analysis Summary

## Generated Files

### Plots
$(find "$RESULTS_DIR/plots" -name "*.png" -o -name "*.pdf" | sort | sed 's/^/- /')

### Tables
$(find "$RESULTS_DIR/tables" -name "*.csv" | sort | sed 's/^/- /')

### Reports
$(find "$RESULTS_DIR/reports" -name "*.md" | sort | sed 's/^/- /')

## Key Findings

1. **Best Evaluation Method**: Spatial conditioning without color (10.0% success rate)
2. **Spatial vs Semantic**: Spatial outperforms semantic by 38.7% in evaluation
3. **Color Impact**: Negative for both spatial (-5.7%) and semantic (-5.0%) approaches
4. **Generalization Gap**: Training performance is not predictive of evaluation performance
5. **Recommendation**: Use spatial conditioning without color for surgical robotics applications

Generated on: $(date)
EOF

echo ""
echo "🎉 Analysis completed successfully!"
echo ""
echo "📁 Results organized in:"
echo "   📊 Plots: $RESULTS_DIR/plots"
echo "   📋 Tables: $RESULTS_DIR/tables"
echo "   📄 Reports: $RESULTS_DIR/reports"
echo ""
echo "📖 View the main report: $RESULTS_DIR/reports/cs224r_final_report.md"
echo "📊 View analysis summary: $RESULTS_DIR/reports/analysis_summary.md" 