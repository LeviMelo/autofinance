import pandas as pd
from pathlib import Path
import sys

# Add the source directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from b3alloc.factors.fama_french_b3 import build_fama_french_factors
from b3alloc.preprocess.align import align_fundamentals_to_prices
from b3alloc.preprocess.returns import compute_returns

def main():
    """
    Main function to pre-compute and save the Fama-French factor panel.
    """
    print("--- Starting Fama-French Factor Construction ---")
    
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed"
    
    if not data_path.exists():
        print(f"ERROR: Processed data directory not found at {data_path}")
        print("Please run 'scripts/update_data.py' first.")
        return

    # --- 1. Load Required Processed Data ---
    print("Loading processed price, fundamental, and risk-free data...")
    try:
        prices_df_long = pd.read_parquet(data_path / "prices_equity_daily.parquet")
        fundamentals_df = pd.read_parquet(data_path / "fundamentals_quarterly.parquet")
        rf_df = pd.read_parquet(data_path / "risk_free_daily.parquet")
        ibov_df = pd.read_parquet(data_path / "index_ibov_daily.parquet")
    except FileNotFoundError as e:
        print(f"ERROR: Missing data file - {e.filename}")
        print("Please ensure 'scripts/update_data.py' has been run successfully.")
        return

    # --- 2. Prepare Data for Factor Model ---
    print("Preparing data panels for factor construction...")
    
    # We need prices in wide format for market cap calculation
    prices_wide = prices_df_long.pivot(index='date', columns='ticker', values='adj_close')
    
    # We need daily fundamentals, correctly lagged
    # A generic lag of 65 trading days (~3 months) is a common heuristic
    # if publish dates aren't available.
    daily_fundamentals = align_fundamentals_to_prices(
        fundamentals_df=fundamentals_df,
        price_dates=prices_wide.index,
        publish_lag_days=65 
    )
    
    # We need simple returns
    all_returns = compute_returns(prices_df_long, rf_df)
    simple_returns_wide = all_returns['simple']
    
    # We need market excess returns for the 'MKT' factor
    market_simple_returns = ibov_df['adj_close'].pct_change()
    market_excess_returns = market_simple_returns - rf_df['rf_daily']
    market_excess_returns = market_excess_returns.reindex(simple_returns_wide.index).ffill().dropna()

    # --- 3. Build Factors ---
    factor_panel = build_fama_french_factors(
        daily_fundamentals_df=daily_fundamentals,
        prices_df=prices_wide,
        returns_df=simple_returns_wide,
        market_excess_returns=market_excess_returns
    )
    
    # --- 4. Save the Factor Panel ---
    output_file = data_path / "factor_panel_daily.parquet"
    factor_panel.to_parquet(output_file)
    
    print(f"\nSuccessfully built and saved Fama-French factor panel to:")
    print(output_file)
    print("\n--- Factor Construction Finished ---")

if __name__ == '__main__':
    main()