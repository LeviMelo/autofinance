import pandas as pd
from pathlib import Path
import logging
import traceback
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from b3alloc.config import load_config
from b3alloc.data import (
    create_equity_price_series,
    create_index_series,
    create_fundamentals_series,
    create_risk_free_series,
    create_fx_series,
)
from b3alloc.preprocess.align import align_fundamentals_to_prices
from b3alloc.preprocess.returns import compute_returns
from b3alloc.factors.fama_french_b3 import build_fama_french_factors
from b3alloc.backtest.engine import BacktestEngine
from b3alloc.backtest.analytics import compute_performance_metrics

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run():
    """Main function to orchestrate the entire backtesting pipeline."""
    try:
        # 1. Load Configuration
        logging.info("--- 1. Loading Configuration ---")
        # Make path relative to the script's location
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "config/config.yaml"
        cfg = load_config(config_path)
        
        # The tickers_file path in YAML is relative to the script's parent (the project root)
        # So, we join the script's parent with the path from config.
        tickers_path = script_dir / cfg.data.tickers_file
        if not tickers_path.exists():
            raise FileNotFoundError(f"Tickers file not found at: {tickers_path}")
        tickers = pd.read_csv(tickers_path)['ticker'].tolist()
        logging.info(f"Loaded {len(tickers)} tickers for backtest.")

        # 2. Data Ingestion
        logging.info("--- 2. Ingesting Data ---")
        prices_long = create_equity_price_series(tickers, cfg.data.start, cfg.data.end)
        index_prices = create_index_series(cfg.data.start, cfg.data.end)
        fundamentals = create_fundamentals_series(tickers)
        risk_free = create_risk_free_series(cfg.data.start, cfg.data.end, series_id=cfg.data.selic_series)

        # 3. Data Preprocessing & Feature Engineering
        logging.info("--- 3. Preprocessing Data ---")
        
        # Calculate returns
        prices_wide = prices_long.pivot(index='date', columns='ticker', values='adj_close')
        returns_bundle = compute_returns(prices_wide, risk_free, benchmark_df=index_prices)
        
        # Align fundamentals
        daily_fundamentals = align_fundamentals_to_prices(
            fundamentals,
            prices_wide.index,
            cfg.data.publish_lag_days
        )
        
        # Calculate market caps
        shares_outstanding_wide = daily_fundamentals.pivot(index='date', columns='ticker', values='shares_outstanding')
        market_caps = prices_wide * shares_outstanding_wide
        
        # Build Fama-French Factors
        logging.info("--- 4. Building Factors ---")
        factors = build_fama_french_factors(
            daily_fundamentals,
            prices_wide,
            returns_bundle['simple'],
            returns_bundle['market_excess']
        )

        # 4. Assemble Data Stores for Backtest Engine
        logging.info("--- 5. Assembling Data Stores ---")
        data_stores = {
            "prices": prices_long,
            "log_returns": returns_bundle['log'],
            "excess_log_returns": returns_bundle['log_excess'],
            "market_excess_returns": returns_bundle['market_excess'],
            "factors": factors,
            "market_caps": market_caps,
        }

        # 5. Run Backtest Engine
        logging.info("--- 6. Running Backtest Engine ---")
        engine = BacktestEngine(cfg, data_stores)
        history_df = engine.run_backtest()

        # 6. Compute and Display Performance Analytics
        logging.info("--- 7. Computing Performance Analytics ---")
        strategy_returns = history_df['portfolio_value'].pct_change().dropna()
        benchmark_returns = returns_bundle['market_simple']
        
        # Align strategy and benchmark returns
        aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        
        performance = compute_performance_metrics(
            strategy_returns=aligned_returns.iloc[:, 0],
            benchmark_returns=aligned_returns.iloc[:, 1],
            risk_free_rate=risk_free['rf_daily'],
            periods_per_year=252 # Daily data
        )
        
        print("\n" + "="*50)
        print("          BACKTEST PERFORMANCE METRICS")
        print("="*50)
        print(performance)
        print("="*50)

    except Exception as e:
        logging.error(f"An error occurred in the pipeline: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    run() 