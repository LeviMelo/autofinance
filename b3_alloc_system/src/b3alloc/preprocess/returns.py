import pandas as pd
import numpy as np

def compute_returns(
    prices_df: pd.DataFrame,
    risk_free_df: pd.DataFrame,
    price_col: str = 'adj_close'
) -> pd.DataFrame:
    """
    Computes daily simple, log, and excess returns for a set of assets.

    The input `prices_df` is expected to be in long format, but is pivoted to
    wide format for efficient vectorized calculations. The output is returned
    in a wide format.

    Args:
        prices_df: A long-format DataFrame of daily equity prices, with columns
                   ['date', 'ticker', 'adj_close', ...].
        risk_free_df: A DataFrame of daily risk-free rates, indexed by date,
                      with a column 'rf_daily'.
        price_col: The column name in `prices_df` to use for returns calculation.
                   Defaults to 'adj_close'.

    Returns:
        A dictionary of DataFrames, each indexed by date with tickers as columns:
        - 'simple': Daily simple returns.
        - 'log': Daily log returns (used for modeling).
        - 'excess_simple': Simple returns minus the daily risk-free rate.
        - 'excess_log': Log returns minus the daily risk-free rate.
    """
    if prices_df.empty or risk_free_df.empty:
        raise ValueError("Input prices_df or risk_free_df cannot be empty.")

    # 1. Pivot prices from long to wide format for vectorized calculations
    # This creates a DataFrame with dates as the index and tickers as columns.
    prices_wide = prices_df.pivot(index='date', columns='ticker', values=price_col)

    # 2. Align risk-free series to the price index
    # This ensures we have a matching risk-free rate for every day of prices.
    rf_aligned, prices_wide = risk_free_df['rf_daily'].align(prices_wide, join='right')
    
    # 3. Calculate simple and log returns
    # The `pct_change()` method is equivalent to (p_t / p_{t-1}) - 1
    simple_returns = prices_wide.pct_change()
    
    # Log returns are calculated as the difference of the natural logarithm of prices
    log_returns = np.log(prices_wide / prices_wide.shift(1))

    # The first row will be NaN after pct_change/shift, which is expected.
    # We will handle this in downstream models.

    # 4. Calculate excess returns
    # We can subtract the risk-free rate series from every column of the returns matrix.
    excess_simple_returns = simple_returns.subtract(rf_aligned, axis=0)
    excess_log_returns = log_returns.subtract(rf_aligned, axis=0)
    
    print("Successfully computed simple, log, and excess returns.")
    
    return {
        'simple': simple_returns,
        'log': log_returns,
        'excess_simple': excess_simple_returns,
        'excess_log': excess_log_returns
    }

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