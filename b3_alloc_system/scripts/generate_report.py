import argparse
import pandas as pd
from pathlib import Path
import json

def create_markdown_report(run_path: Path) -> str:
    """
    Generates a full Markdown report from a backtest run's artifacts.

    Args:
        run_path: The Path object pointing to the specific run directory
                  (e.g., 'reports/runs/run_20250718_103000').

    Returns:
        A string containing the complete Markdown report.
    """
    print(f"Generating Markdown report for run: {run_path.name}")
    
    # --- 1. Load Artifacts ---
    try:
        metrics = json.loads((run_path / "performance_summary.json").read_text())
        config_text = (run_path / "config.yaml").read_text()
    except FileNotFoundError:
        return f"# Error: Could not generate report for {run_path.name}.\nMissing 'performance_summary.json' or 'config.yaml'."

    # --- 2. Build Markdown Sections ---
    
    # Header
    header = f"# Backtest Report: {run_path.name}\n\n"
    header += f"This report details the performance of a strategy run executed on `{run_path.name.split('_')[1]}`.\n"
    
    # Summary Table
    summary_table = "## 1. Summary Performance\n\n"
    summary_table += "| Metric                  | Strategy | Benchmark |\n"
    summary_table += "| ----------------------- | -------- | --------- |\n"
    summary_table += f"| **CAGR**                | `{metrics.get('CAGR', 0):.2%}` | `{metrics.get('Benchmark CAGR', 0):.2%}` |\n"
    summary_table += f"| **Annualized Volatility** | `{metrics.get('Annualized Volatility', 0):.2%}` | `{metrics.get('Benchmark Volatility', 0):.2%}` |\n"
    summary_table += f"| **Sharpe Ratio**          | `{metrics.get('Sharpe Ratio', 0):.2f}` | *N/A* |\n"
    summary_table += f"| **Max Drawdown**          | `{metrics.get('Max Drawdown', 0):.2%}` | *N/A* |\n"
    summary_table += f"| **Alpha (annualized)**    | `{metrics.get('Alpha (annualized)', 0):.2%}` | *N/A* |\n"
    summary_table += f"| **Beta**                  | `{metrics.get('Beta', 0):.2f}` | *N/A* |\n\n"

    # Plots
    plots = "## 2. Visualizations\n\n"
    plots += "### Equity Curve (Log Scale)\n"
    plots += "![Equity Curve](./equity_curve.png)\n\n"
    plots += "### Drawdowns\n"
    plots += "![Drawdowns](./drawdowns.png)\n\n"
    
    # Conditionally include the weights plot if it exists
    if (run_path / "weights_evolution.png").exists():
        plots += "### Portfolio Weights Evolution\n"
        plots += "![Weights Evolution](./weights_evolution.png)\n\n"
    
    # Configuration
    config_section = "## 3. Configuration\n\n"
    config_section += "This run was executed with the following configuration:\n"
    config_section += f"```yaml\n{config_text}\n```\n"

    # Combine all parts
    full_report = header + summary_table + plots + config_section
    
    return full_report

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generates a Markdown report from a backtest run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the backtest run directory (e.g., reports/runs/run_...).",
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    run_path = project_root / args.run_dir
    
    if not run_path.is_dir():
        print(f"ERROR: Run directory not found at '{run_path}'")
        return
        
    # Generate the report content
    markdown_content = create_markdown_report(run_path)
    
    # Save the report to a file
    report_file_path = run_path / "report.md"
    report_file_path.write_text(markdown_content, encoding='utf-8')
    
    print(f"\nSuccessfully generated and saved report to:")
    print(report_file_path)

if __name__ == '__main__':
    # To test this script, you would first need to run `scripts/run_backtest.py`
    # which creates the necessary artifacts in a 'reports/runs/run_...' directory.
    # Then, you would execute this script from the command line:
    # python scripts/generate_report.py --run_dir reports/runs/run_20240101_120000
    
    print("--- Report Generation Script ---")
    print("This script is intended to be run from the command line after a backtest.")
    print("Example usage:")
    print("python scripts/generate_report.py --run_dir <path_to_your_run_directory>")