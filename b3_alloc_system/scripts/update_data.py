import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add the source directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from b3alloc.data.ingest_selic import create_risk_free_series
from b3alloc.data.ingest_prices import create_equity_price_series, create_index_series
from b3alloc.data.ingest_fundamentals import create_fundamentals_series
from b3alloc.data.ingest_fx import create_fx_series
from b3alloc.config import load_config

# --- Configuration ---
# The universe of tickers is now loaded from the config file.
IBOV_TICKER = "^BVSP"

def main():
    """
    Main function to run the entire data ingestion and processing pipeline.
    """
    print("--- Starting Data Update and Processing ---")
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Load configuration to get the universe and date ranges
    # This assumes a default or specified config file. For simplicity, we point to one.
    # A more robust CLI would allow specifying which portfolio's data to update.
    try:
        config_path = project_root / 'config' / 'portfolio_A.yaml'
        cfg = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        print("ERROR: Default config 'config/portfolio_A.yaml' not found.")
        print("Please create a config file for your portfolio.")
        return

    output_path = project_root / "data" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Data will be saved to: {output_path}")

    # --- Step 1: Ingest SELIC (Risk-Free Rate) ---
    try:
        print("\n[1/5] Fetching and processing SELIC risk-free rate...")
        risk_free_df = create_risk_free_series(cfg.data.start, cfg.data.end)
        risk_free_df.to_parquet(output_path / "risk_free_daily.parquet")
        print("  -> Saved 'risk_free_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching SELIC data: {e}")

    # --- Step 2: Ingest IBOVESPA Index Prices ---
    try:
        print(f"\n[2/5] Fetching and processing benchmark index ({IBOV_TICKER})...")
        index_df = create_index_series(cfg.data.start, cfg.data.end, index_ticker=IBOV_TICKER)
        index_df.to_parquet(output_path / "index_ibov_daily.parquet")
        print(f"  -> Saved 'index_ibov_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching index data: {e}")

    # --- Step 3: Ingest Equity Prices ---
    try:
        print("\n[3/5] Fetching and processing equity prices...")
        # Load tickers from the config's data section
        ticker_list = pd.read_csv(project_root / cfg.data.tickers_file)['ticker'].tolist()
        equity_prices_df = create_equity_price_series(ticker_list, cfg.data.start, cfg.data.end)
        equity_prices_df.to_parquet(output_path / "prices_equity_daily.parquet")
        print("  -> Saved 'prices_equity_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching equity prices: {e}")

    # --- Step 4: Ingest Fundamentals Data ---
    try:
        print("\n[4/5] Fetching and processing quarterly fundamentals...")
        fundamentals_df = create_fundamentals_series(ticker_list)
        fundamentals_df.to_parquet(output_path / "fundamentals_quarterly.parquet")
        print("  -> Saved 'fundamentals_quarterly.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching fundamentals: {e}")

    # --- Step 5: Ingest FX Data ---
    try:
        print("\n[5/5] Fetching and processing FX (USD/BRL) data...")
        fx_df = create_fx_series(cfg.data.start, cfg.data.end, series_id=cfg.universe.fx_series_id)
        fx_df.to_parquet(output_path / "fx_usd_brl_daily.parquet")
        print("  -> Saved 'fx_usd_brl_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching FX data: {e}")
        
    print("\n--- Data Update and Processing Finished ---")

if __name__ == '__main__':
    # To run this script, ensure you have an active internet connection
    # and have installed all dependencies from environment.yml.
    # Execute from the project root directory: `python scripts/update_data.py`
    main()
