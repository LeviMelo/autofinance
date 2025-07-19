import pandas as pd
from typing import List, Dict

def flag_price_outliers(
    returns_df: pd.DataFrame,
    window: int = 63,
    z_score_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Flags extreme daily returns based on a rolling Z-score.

    This function helps identify potential data errors or extreme events.
    It does not modify the data but adds boolean flag columns.

    Args:
        returns_df: A wide-format DataFrame of daily returns (tickers as columns).
        window: The rolling window length in trading days (e.g., 63 for ~3 months).
        z_score_threshold: The number of standard deviations to use as the outlier cutoff.

    Returns:
        A DataFrame of the same shape as input, with boolean flags where
        returns are considered outliers.
    """
    rolling_mean = returns_df.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = returns_df.rolling(window=window, min_periods=window // 2).std()

    # Calculate Z-scores, handling potential division by zero
    z_scores = (returns_df - rolling_mean) / (rolling_std + 1e-9)
    
    outlier_flags = z_scores.abs() > z_score_threshold
    
    num_outliers = outlier_flags.sum().sum()
    print(f"Flagged {num_outliers} return outliers (Z-score > {z_score_threshold}).")
    
    return outlier_flags

def get_liquid_universe(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback_days: int,
    min_trading_ratio: float = 0.90,
    min_avg_volume: int = 1000000
) -> List[str]:
    """
    Filters the asset universe based on liquidity criteria at a specific rebalance date.

    As per the specification, this function ensures that only assets with
    sufficient trading activity and history are included in the portfolio
    construction for a given period.

    Args:
        prices_df: A wide-format DataFrame of daily prices (tickers as columns).
        volume_df: A wide-format DataFrame of daily volumes.
        rebalance_date: The date on which the universe is being screened.
        lookback_days: The number of trading days to look back for evaluation.
        min_trading_ratio: The minimum fraction of days an asset must have a
                           valid price in the lookback window.
        min_avg_volume: The minimum average daily trading volume (in BRL, assuming
                        price * volume).

    Returns:
        A list of ticker symbols that are eligible for inclusion in the model.
    """
    # 1. Define the lookback window
    lookback_start_date = rebalance_date - pd.Timedelta(days=lookback_days)
    
    prices_window = prices_df.loc[lookback_start_date:rebalance_date]
    volume_window = volume_df.loc[lookback_start_date:rebalance_date]

    if prices_window.empty or volume_window.empty:
        print("Warning: Not enough data in the lookback window to determine universe.")
        return []

    # 2. Minimum History / Trading Day Ratio
    # Count non-NaN prices for each asset in the window
    trading_days_count = prices_window.notna().sum()
    required_trading_days = len(prices_window) * min_trading_ratio
    
    traded_enough = trading_days_count >= required_trading_days
    
    # 3. Minimum Average Volume
    # Calculate average daily traded value (price * volume)
    # Using the average price over the window to be robust to price changes
    avg_price = prices_window.mean()
    avg_volume = volume_window.mean()
    avg_traded_value = avg_price * avg_volume
    
    enough_volume = avg_traded_value >= min_avg_volume
    
    # 4. Combine filters
    eligible_tickers = traded_enough & enough_volume
    
    liquid_universe = eligible_tickers[eligible_tickers].index.tolist()
    
    total_assets = len(eligible_tickers)
    print(f"Screened universe on {rebalance_date.date()}: {len(liquid_universe)} of {total_assets} assets passed liquidity filters.")
    
    return liquid_universe


if __name__ == '__main__':
    print("--- Running Clean Module Standalone Test ---")

    # 1. Create dummy input data
    dates = pd.to_datetime(pd.date_range("2023-01-01", "2023-06-30", freq="B"))
    
    prices_data = {
        # Liquid asset, always trades
        'LIQUID_A': np.linspace(100, 110, len(dates)),
        # Illiquid asset, many missing prices
        'ILLIQUID_B': np.linspace(50, 55, len(dates)),
        # Low volume asset
        'LOWVOL_C': np.linspace(20, 22, len(dates)),
        # Asset with a huge price spike (outlier)
        'OUTLIER_D': np.linspace(30, 33, len(dates)),
    }
    
    # Introduce NaNs for the illiquid asset
    prices_data['ILLIQUID_B'][10:40] = np.nan
    # Introduce an outlier
    prices_data['OUTLIER_D'][50] = 50 
    
    dummy_prices = pd.DataFrame(prices_data, index=dates)

    volume_data = {
        'LIQUID_A': [200000] * len(dates),
        'ILLIQUID_B': [150000] * len(dates),
        'LOWVOL_C': [10000] * len(dates), # Avg value will be ~20*10000 = 200k, below threshold
        'OUTLIER_D': [180000] * len(dates),
    }
    dummy_volume = pd.DataFrame(volume_data, index=dates)


    # --- Test 1: Outlier Flagging ---
    print("\n--- Testing flag_price_outliers ---")
    try:
        dummy_returns = dummy_prices.pct_change()
        outlier_flags = flag_price_outliers(dummy_returns, window=20, z_score_threshold=3.0)
        
        print("\nOutlier flags (showing only non-zero rows):")
        print(outlier_flags[outlier_flags.any(axis=1)])
        
        # Validation
        assert outlier_flags.loc['2023-03-15', 'OUTLIER_D'], "Expected outlier was not flagged!"
        print("\nOK: Outlier correctly identified.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- Test 2: Liquidity Filtering ---
    print("\n--- Testing get_liquid_universe ---")
    try:
        rebalance_date = pd.to_datetime("2023-06-30")
        lookback_days = 180
        
        liquid_assets = get_liquid_universe(
            prices_df=dummy_prices,
            volume_df=dummy_volume,
            rebalance_date=rebalance_date,
            lookback_days=lookback_days,
            min_trading_ratio=0.85,  # 85% of days must have price
            min_avg_volume=1000000 # 1 Million BRL
        )
        
        print(f"\nEligible assets on {rebalance_date.date()}: {liquid_assets}")
        
        # Validation
        # LIQUID_A: Should pass (105 * 200k > 1M)
        # ILLIQUID_B: Should fail (too many NaNs)
        # LOWVOL_C: Should fail (avg value ~21*10k = 210k < 1M)
        # OUTLIER_D: Should pass (avg value ~31*180k > 1M)
        expected_universe = ['LIQUID_A', 'OUTLIER_D']
        assert set(liquid_assets) == set(expected_universe), f"Expected {expected_universe} but got {liquid_assets}"
        print("\nOK: Universe filtering works as expected.")
        
    except Exception as e:
        print(f"An error occurred: {e}")