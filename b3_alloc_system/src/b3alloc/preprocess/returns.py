import pandas as pd
import numpy as np

def compute_returns(
    prices_df: pd.DataFrame,
    risk_free_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    price_col: str = 'adj_close'
) -> pd.DataFrame:
    """
    Computes various return series from wide-format price data.

    Args:
        prices_df: Wide-format DataFrame of daily prices, index=date, columns=ticker.
        risk_free_df: DataFrame of daily and annualized risk-free rates.
        benchmark_df: DataFrame of the benchmark index price series.
        price_col: The column to use for price data (default: 'adj_close').

    Returns:
        A dictionary of DataFrames containing simple, log, excess simple,
        and excess log returns, plus market returns.
    """
    
    # The input prices_df is already in wide format, no pivot needed.
    prices_wide = prices_df

    # --- Equity Returns ---
    simple_returns = prices_wide.pct_change()
    log_returns = np.log(prices_wide / prices_wide.shift(1))
    
    # Align risk-free rate to the returns index
    rf_daily = risk_free_df['rf_daily'].reindex(simple_returns.index).ffill()
    
    # Calculate excess returns
    excess_simple_returns = simple_returns.subtract(rf_daily, axis=0)
    excess_log_returns = log_returns.subtract(rf_daily, axis=0)
    
    # --- Benchmark Market Returns ---
    market_simple_returns = benchmark_df[price_col].pct_change()
    market_log_returns = np.log(benchmark_df[price_col] / benchmark_df[price_col].shift(1))
    market_excess_returns = market_simple_returns - rf_daily
    
    returns_bundle = {
        "simple": simple_returns.dropna(how='all'),
        "log": log_returns.dropna(how='all'),
        "simple_excess": excess_simple_returns.dropna(how='all'),
        "log_excess": excess_log_returns.dropna(how='all'),
        "market_simple": market_simple_returns.dropna(),
        "market_log": market_log_returns.dropna(),
        "market_excess": market_excess_returns.dropna()
    }
    
    print("Successfully computed all return series.")
    return returns_bundle

if __name__ == '__main__':
    # Standalone test for the returns module
    print("--- Running Returns Module Standalone Test ---")

    # 1. Create dummy input data
    dates = pd.to_datetime(pd.date_range("2023-01-01", "2023-01-10", freq="B"))
    
    dummy_prices_long = pd.DataFrame({
        'date': dates.repeat(2),
        'ticker': ['TICKER_A', 'TICKER_B'] * len(dates),
        'adj_close': [
            100, 200, 101, 200, 102, 199, 103, 201, 102, 202,
            104, 203, 105, 202
        ]
    })
    
    dummy_rf_daily = pd.DataFrame({
        'rf_daily': [0.0005] * len(dates) # 0.05% per day
    }, index=dates)

    print("--- Input Data ---")
    print("Long Prices:")
    print(dummy_prices_long)
    print("\nDaily Risk-Free Rate:")
    print(dummy_rf_daily)

    # 2. Run the computation function
    try:
        returns_dict = compute_returns(dummy_prices_long, dummy_rf_daily)

        print("\n--- Output ---")
        
        print("\nLog Returns (for GARCH/VAR modeling):")
        print(returns_dict['log'].head())

        print("\nExcess Simple Returns (for performance calculation):")
        print(returns_dict['excess_simple'].head())

        # 3. Validation
        # Manual check for TICKER_A on 2023-01-03
        # Price change from 100 to 101
        expected_log_return = np.log(101/100)
        actual_log_return = returns_dict['log'].loc['2023-01-03', 'TICKER_A']
        
        expected_excess_simple = (101/100 - 1) - 0.0005
        actual_excess_simple = returns_dict['excess_simple'].loc['2023-01-03', 'TICKER_A']
        
        print("\n--- Validation ---")
        print(f"Expected Log Return for TICKER_A on 2023-01-03: {expected_log_return:.6f}")
        print(f"Actual Log Return: {actual_log_return:.6f}")
        assert np.isclose(expected_log_return, actual_log_return), "Log return mismatch!"

        print(f"Expected Excess Simple Return for TICKER_A on 2023-01-03: {expected_excess_simple:.6f}")
        print(f"Actual Excess Simple Return: {actual_excess_simple:.6f}")
        assert np.isclose(expected_excess_simple, actual_excess_simple), "Excess simple return mismatch!"
        
        print("\nOK: Test calculations match expected values.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()