import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

def plot_equity_curve(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Figure:
    """
    Plots the equity curve (cumulative returns) for the strategy vs. a benchmark.

    Args:
        strategy_returns: A pandas Series of the strategy's periodic returns.
        benchmark_returns: A pandas Series of the benchmark's periodic returns.

    Returns:
        A matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    ax.plot(strategy_cumulative.index, strategy_cumulative, label="Strategy", color="blue", lw=2)
    ax.plot(benchmark_cumulative.index, benchmark_cumulative, label="Benchmark", color="gray", ls='--')
    
    # Use a log scale for the y-axis to better visualize relative performance
    ax.set_yscale('log')
    
    ax.set_title('Equity Curve (Strategy vs. Benchmark)', fontsize=16)
    ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Format dates on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    return fig

def plot_drawdowns(strategy_returns: pd.Series) -> Figure:
    """
    Plots the drawdown periods for the strategy.

    Args:
        strategy_returns: A pandas Series of the strategy's periodic returns.

    Returns:
        A matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', lw=1)
    
    ax.set_title('Portfolio Drawdowns', fontsize=16)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    return fig

def plot_weights_evolution(weights_df: pd.DataFrame) -> Figure:
    """
    Plots the evolution of asset weights over time as a stacked area chart.

    Args:
        weights_df: A DataFrame where the index is the date and columns are
                    tickers, containing portfolio weights at each rebalance.

    Returns:
        A matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use pandas' built-in plotting for a stacked area chart
    weights_df.plot.area(ax=ax, stacked=True, lw=0)
    
    ax.set_title('Portfolio Weights Evolution', fontsize=16)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylim(0, 1) # Weights should sum to 1
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Improve legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title='Assets')
    
    fig.tight_layout() # Adjust layout to make room for legend
    
    return fig


if __name__ == '__main__':
    print("--- Running Visualization Module Standalone Test ---")
    
    # --- GIVEN ---
    # Create synthetic data for plotting
    periods = 12 * 5 # 5 years of monthly data
    dates = pd.date_range("2018-01-01", periods=periods, freq="MS")
    
    bench_ret = pd.Series(
        np.random.randn(periods) * 0.05 + 0.006, index=dates
    )
    strat_ret = pd.Series(
        0.8 * bench_ret + np.random.randn(periods) * 0.03 + 0.004, index=dates
    )
    
    # Create a dummy weights dataframe
    tickers = ['Asset A', 'Asset B', 'Asset C', 'Asset D']
    weights = np.random.rand(len(dates), len(tickers))
    weights = weights / weights.sum(axis=1, keepdims=True)
    weights_df = pd.DataFrame(weights, index=dates, columns=tickers)

    print("\nGenerating plots from synthetic data...")
    
    # --- WHEN & THEN ---
    try:
        # 1. Test Equity Curve Plot
        fig1 = plot_equity_curve(strat_ret, bench_ret)
        fig1.suptitle("Test: Equity Curve", y=1.02)
        
        # 2. Test Drawdown Plot
        fig2 = plot_drawdowns(strat_ret)
        fig2.suptitle("Test: Drawdowns", y=1.02)

        # 3. Test Weights Evolution Plot
        fig3 = plot_weights_evolution(weights_df)
        fig3.suptitle("Test: Weights Evolution", y=1.02)
        
        print("\nOK: All plot functions executed without error.")
        print("Displaying generated plots...")
        plt.show()

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()