import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add the source directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from b3alloc.data.ingest_selic import create_risk_free_series
from b3alloc.data.ingest_prices import create_equity_price_series, create_index_series
from b3alloc.data.ingest_fundamentals import create_fundamentals_series

# --- Configuration ---
# For this PoC, we define the universe of tickers directly here.
# This can be moved to a CSV file as suggested in the project spec.
UNIVERSE_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "WEGE3.SA", "MGLU3.SA", "LREN3.SA", "B3SA3.SA", "SUZB3.SA",
    "GGBR4.SA", "BBAS3.SA", "ITSA4.SA", "RENT3.SA", "ELET3.SA",
    # Add a BDR for FX testing
    "AAPL34.SA"
]
IBOV_TICKER = "^BVSP"

# Define the historical period for data download
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def main():
    """
    Main function to run the entire data ingestion and processing pipeline.
    """
    print("--- Starting Data Update and Processing ---")
    
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Data will be saved to: {output_path}")

    # --- Step 1: Ingest SELIC (Risk-Free Rate) ---
    try:
        print("\n[1/4] Fetching and processing SELIC risk-free rate...")
        risk_free_df = create_risk_free_series(START_DATE, END_DATE)
        risk_free_df.to_parquet(output_path / "risk_free_daily.parquet")
        print("  -> Saved 'risk_free_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching SELIC data: {e}")

    # --- Step 2: Ingest IBOVESPA Index Prices ---
    try:
        print(f"\n[2/4] Fetching and processing benchmark index ({IBOV_TICKER})...")
        index_df = create_index_series(START_DATE, END_DATE, index_ticker=IBOV_TICKER)
        index_df.to_parquet(output_path / "index_ibov_daily.parquet")
        print(f"  -> Saved 'index_ibov_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching index data: {e}")

    # --- Step 3: Ingest Equity Prices ---
    try:
        print("\n[3/4] Fetching and processing equity prices...")
        equity_prices_df = create_equity_price_series(UNIVERSE_TICKERS, START_DATE, END_DATE)
        equity_prices_df.to_parquet(output_path / "prices_equity_daily.parquet")
        print("  -> Saved 'prices_equity_daily.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching equity prices: {e}")

    # --- Step 4: Ingest Fundamentals Data ---
    try:
        print("\n[4/4] Fetching and processing quarterly fundamentals...")
        fundamentals_df = create_fundamentals_series(UNIVERSE_TICKERS)
        fundamentals_df.to_parquet(output_path / "fundamentals_quarterly.parquet")
        print("  -> Saved 'fundamentals_quarterly.parquet'")
    except Exception as e:
        print(f"  -> ERROR fetching fundamentals: {e}")
        
    print("\n--- Data Update and Processing Finished ---")

if __name__ == '__main__':
    # To run this script, ensure you have an active internet connection
    # and have installed all dependencies from environment.yml.
    # Execute from the project root directory: `python scripts/update_data.py`
    main()
