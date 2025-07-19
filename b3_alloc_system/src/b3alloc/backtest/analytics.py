import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict

def _calculate_max_drawdown(cumulative_returns: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Helper function to calculate the maximum drawdown and its dates."""
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    
    max_dd = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = cumulative_returns.index[cumulative_returns.index <= end_date] \
                                   [np.argmax(cumulative_returns.loc[:end_date].values)]
    
    return max_dd, start_date, end_date

def compute_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: pd.Series,
    periods_per_year: int
) -> pd.Series:
    """
    Computes a comprehensive set of performance and risk analytics for a strategy.

    Args:
        strategy_returns: A pandas Series of the strategy's periodic returns.
        benchmark_returns: A pandas Series of the benchmark's periodic returns.
        risk_free_rate: A pandas Series of the risk-free rate for the same periods.
        periods_per_year: The number of return periods in a year (e.g., 12 for monthly).

    Returns:
        A pandas Series containing all calculated performance metrics.
    """
    # --- Data Alignment ---
    data = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns,
        'risk_free': risk_free_rate
    }).dropna()
    
    # --- Cumulative Returns ---
    strategy_cumulative = (1 + data['strategy']).cumprod()
    benchmark_cumulative = (1 + data['benchmark']).cumprod()
    total_return_strategy = strategy_cumulative.iloc[-1] - 1
    total_return_benchmark = benchmark_cumulative.iloc[-1] - 1
    
    # --- Annualized Metrics ---
    num_years = len(data) / periods_per_year
    cagr_strategy = (1 + total_return_strategy)**(1/num_years) - 1
    cagr_benchmark = (1 + total_return_benchmark)**(1/num_years) - 1
    
    vol_strategy = data['strategy'].std() * np.sqrt(periods_per_year)
    vol_benchmark = data['benchmark'].std() * np.sqrt(periods_per_year)
    
    # --- Risk-Adjusted Returns ---
    annualized_rf = data['risk_free'].mean() * periods_per_year
    sharpe_ratio = (cagr_strategy - annualized_rf) / vol_strategy
    
    # Sortino Ratio
    downside_returns = data['strategy'][data['strategy'] < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = (cagr_strategy - annualized_rf) / downside_deviation if downside_deviation > 0 else np.inf
    
    # --- Drawdown Analysis ---
    max_dd_strategy, dd_start, dd_end = _calculate_max_drawdown(strategy_cumulative)
    calmar_ratio = cagr_strategy / abs(max_dd_strategy)
    
    # --- Regression-Based Metrics (Alpha and Beta) ---
    strategy_excess_ret = data['strategy'] - data['risk_free']
    benchmark_excess_ret = data['benchmark'] - data['risk_free']
    
    X = sm.add_constant(benchmark_excess_ret)
    Y = strategy_excess_ret
    model = sm.OLS(Y, X).fit()
    
    beta = model.params['benchmark']
    # Annualize the alpha (which is a per-period intercept)
    alpha = model.params['const'] * periods_per_year
    
    # Information Ratio (measures consistency of alpha)
    tracking_error = (strategy_excess_ret - benchmark_excess_ret).std() * np.sqrt(periods_per_year)
    information_ratio = alpha / tracking_error if tracking_error > 0 else np.inf
    
    metrics = {
        'Total Return': total_return_strategy,
        'CAGR': cagr_strategy,
        'Annualized Volatility': vol_strategy,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_dd_strategy,
        'Calmar Ratio': calmar_ratio,
        'Alpha (annualized)': alpha,
        'Beta': beta,
        'Information Ratio': information_ratio,
        'Skewness': data['strategy'].skew(),
        'Kurtosis': data['strategy'].kurtosis(),
        'Benchmark CAGR': cagr_benchmark,
        'Benchmark Volatility': vol_benchmark
    }
    
    return pd.Series(metrics, name='Performance Metrics')


if __name__ == '__main__':
    print("--- Running Analytics Module Standalone Test ---")

    # --- GIVEN ---
    # Create synthetic data where the strategy clearly outperforms the benchmark
    # with lower volatility and some generated alpha.
    periods = 12 * 5 # 5 years of monthly data
    periods_per_year = 12
    dates = pd.date_range("2018-01-01", periods=periods, freq="MS")
    
    # Benchmark has a 8% CAGR, 20% vol
    bench_ret = np.random.randn(periods) * (0.20 / np.sqrt(periods_per_year)) + (0.08 / periods_per_year)
    
    # Strategy has a 12% CAGR, 15% vol, and an alpha component
    # alpha = 0.04/12 per month; beta = 0.8
    alpha_monthly = 0.04 / periods_per_year
    beta_sim = 0.8
    strat_ret = alpha_monthly + beta_sim * bench_ret + np.random.randn(periods) * (0.05 / np.sqrt(periods_per_year))
    
    # Convert to Series
    strategy_returns = pd.Series(strat_ret, index=dates)
    benchmark_returns = pd.Series(bench_ret, index=dates)
    risk_free_rate = pd.Series(0.02 / periods_per_year, index=dates)

    # --- WHEN ---
    try:
        performance_summary = compute_performance_metrics(
            strategy_returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        
        # --- THEN ---
        print("\n--- Performance Summary ---")
        print(performance_summary)
        
        # --- Validation ---
        print("\n--- Validation ---")
        assert performance_summary['CAGR'] > performance_summary['Benchmark CAGR']
        print("OK: Strategy CAGR > Benchmark CAGR.")
        
        assert performance_summary['Annualized Volatility'] < performance_summary['Benchmark Volatility']
        print("OK: Strategy Volatility < Benchmark Volatility.")
        
        assert performance_summary['Sharpe Ratio'] > 0.5
        print("OK: Sharpe Ratio is in a reasonable range for a good strategy.")
        
        # Check if estimated alpha and beta are close to the simulation parameters
        assert abs(performance_summary['Alpha (annualized)'] - 0.04) < 0.015
        print(f"OK: Estimated Alpha ({performance_summary['Alpha (annualized)']:.4f}) is close to simulated alpha (0.04).")
        
        assert abs(performance_summary['Beta'] - beta_sim) < 0.1
        print(f"OK: Estimated Beta ({performance_summary['Beta']:.4f}) is close to simulated beta ({beta_sim}).")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()