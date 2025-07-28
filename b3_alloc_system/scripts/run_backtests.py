import argparse
import pandas as pd
from pathlib import Path
import shutil
import yaml
from datetime import datetime
import sys
import numpy as np
import cvxpy as cp
from typing import Dict

# --- System Path Setup ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Import All Necessary Modules from the Library ---
from b3alloc.config import load_config
from b3alloc.utils_dates import generate_rebalance_dates
from b3alloc.preprocess.clean import get_liquid_universe
from b3alloc.risk.risk_engine import build_covariance_matrix
from b3alloc.returns.ff_view import create_fama_french_view
from b3alloc.returns.var_view import create_var_view
from b3alloc.bl.black_litterman import calculate_risk_aversion, calculate_equilibrium_returns, calculate_posterior_returns
from b3alloc.bl.view_builder import build_full_view_matrices
from b3alloc.bl.confidence import estimate_view_uncertainty
from b3alloc.optimize.mean_variance import run_mean_variance_optimization
from b3alloc.optimize.constraints import build_optimizer_constraints
from b3alloc.trades.trade_calculator import calculate_target_shares, resolve_fractional_shares, compute_trade_list
from b3alloc.backtest.portfolio_accounting import PortfolioLedger
from b3alloc.taxes.ledger import TaxLedger
from b3alloc.taxes.tax_tracker import calculate_monthly_taxes
from b3alloc.taxes.darf_reporter import generate_darf_excel_report
from b3alloc.backtest.analytics import compute_performance_metrics
from b3alloc.viz.plots_portfolio import plot_equity_curve, plot_drawdowns
from b3alloc.views.views_parser import parse_qualitative_views
from b3alloc.preprocess.align import align_fundamentals_to_prices

import warnings

# Suppress specific warnings that are polluting the output
# --- FIXED: Correctly suppress warnings without causing a NameError ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
# The following is a more robust way to filter the specific ValueWarning from statsmodels
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.")
# ----------------------------------------------------------------------


def _prepare_data_stores(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Loads all necessary data from processed files into a dictionary."""
    print("Loading all processed data stores...")
    stores = {}
    
    required_files = list(data_path.glob("*.parquet"))
    if not required_files:
        raise FileNotFoundError(f"No .parquet files found in {data_path}. Please run data ingestion scripts.")

    for f_path in required_files:
        try:
            key = f_path.stem
            stores[key] = pd.read_parquet(f_path)
        except Exception as e:
            print(f"Warning: Could not load or process {f_path.name}. Error: {e}")

    if 'prices_equity_daily' in stores:
        prices_long = stores['prices_equity_daily']
        stores['prices_wide'] = prices_long.pivot(index='date', columns='ticker', values='adj_close')
        stores['volume_wide'] = prices_long.pivot(index='date', columns='ticker', values='volume')
        
        if 'fundamentals_quarterly' in stores:
            print("Aligning fundamentals to calculate daily market caps...")
            PUBLISH_LAG_DAYS = 90
            
            daily_fundamentals = align_fundamentals_to_prices(
                fundamentals_df=stores['fundamentals_quarterly'],
                price_dates=stores['prices_wide'].index,
                publish_lag_days=PUBLISH_LAG_DAYS
            )
            
            if not daily_fundamentals.empty:
                shares_wide = daily_fundamentals.pivot(index='date', columns='ticker', values='shares_outstanding')
                aligned_prices, aligned_shares = stores['prices_wide'].align(shares_wide, join='left', axis=0)
                aligned_shares = aligned_shares.ffill()
                
                stores['market_caps'] = (aligned_prices * aligned_shares).dropna(how='all')
                print("Successfully calculated daily market caps.")
            else:
                print("Warning: Daily fundamentals alignment produced no data. Market caps cannot be calculated.")

    if 'prices_wide' in stores and 'risk_free_daily' in stores:
        rf_series = stores['risk_free_daily']['rf_daily']
        stores['simple_returns'] = stores['prices_wide'].pct_change()
        
        if 'index_ibov_daily' in stores:
            stores['market_simple_returns'] = stores['index_ibov_daily']['adj_close'].pct_change()
            stores['market_excess_returns'] = (stores['market_simple_returns'] - rf_series).dropna()

    if 'prices_wide' in stores:
        stores['log_returns'] = np.log(1 + stores['prices_wide'].pct_change()).dropna(how='all', axis=0)
    
    print("Data stores prepared successfully.")
    return stores

def main():
    parser = argparse.ArgumentParser(description="Run the full portfolio allocation backtest.")
    parser.add_argument("--config", type=str, required=True, help="Path to the master YAML configuration file.")
    args = parser.parse_args()

    # --- 1. Setup ---
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config
    cfg = load_config(config_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "reports" / "runs" / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_path = project_root / "data" / "processed"
    data_stores = _prepare_data_stores(data_path)

    # --- 2. Initialization ---
    p_ledger = PortfolioLedger(initial_capital=cfg.backtest.initial_capital)
    tax_ledger = TaxLedger()
    rebalance_dates = generate_rebalance_dates(cfg.backtest.start, cfg.backtest.end, cfg.backtest.rebalance)

    # --- 3. Main Backtesting Loop ---
    for rebalance_date in rebalance_dates:
        print(f"\n--- Processing Rebalance Date: {rebalance_date.date()} ---")
        
        # --- A. Get Current State & Universe ---
        current_prices = data_stores['prices_wide'].loc[rebalance_date]
        portfolio_value = p_ledger.get_portfolio_value(current_prices, rebalance_date)
        
        lookback_start = rebalance_date - pd.DateOffset(years=cfg.backtest.lookback_years)
        universe = get_liquid_universe(
            data_stores['prices_wide'].loc[lookback_start:rebalance_date],
            data_stores['volume_wide'].loc[lookback_start:rebalance_date],
            rebalance_date, lookback_days=365
        )
        if len(universe) < 2: continue

        # --- B. Run Full Model Pipeline ---
        try:
            # Slice data for models
            returns_slice = data_stores['log_returns'].loc[lookback_start:rebalance_date, universe]
            factors_slice = data_stores['factor_panel_daily'].loc[lookback_start:rebalance_date]
            fx_slice = data_stores['fx_usd_brl_daily'][['USD_BRL_log_return']].loc[lookback_start:rebalance_date]
            
            # Risk Model (FX-aware) 
            risk_config_dict = cfg.risk_engine.model_dump()
            sigma, risk_diags = build_covariance_matrix(returns_slice, risk_config_dict, fx_returns_df=fx_slice)
            final_universe = sorted(sigma.columns.tolist())
            returns_slice = returns_slice[final_universe]
            
            # Return Models (FX-aware)
            asset_flags = cfg.universe.asset_flags if hasattr(cfg.universe, 'asset_flags') else {}
            (ff_betas, ff_view), ff_diags = create_fama_french_view(
                returns_slice, factors_slice, cfg.return_engine.factor,
                fx_returns_df=fx_slice, asset_fx_sensitivity=asset_flags
            )
            var_view, var_diags = create_var_view(returns_slice, cfg.return_engine.var)
            model_views = {'ff_view': ff_view, 'var_view': var_view}
            
            # Parse Qualitative views
            qual_views, P_qual, Q_qual = None, None, None
            if cfg.black_litterman.qualitative_views_file:
                qual_views_path = project_root / cfg.black_litterman.qualitative_views_file
                if qual_views_path.exists():
                    qual_views = yaml.safe_load(qual_views_path.read_text())['views']
                    P_qual, Q_qual = parse_qualitative_views(qual_views, final_universe)
                else:
                    print(f"Warning: Qualitative views file not found at {qual_views_path}")
            
            # Build the FULL P and Q matrices
            P, Q = build_full_view_matrices(model_views, qual_views, final_universe)
            if P is None: raise ValueError("No valid views could be constructed.")

            # Estimate Uncertainty (Omega)
            view_diags = {'ff_view': ff_diags, 'var_view': var_diags}
            Omega = estimate_view_uncertainty(
                model_views=model_views,
                qualitative_views=qual_views,
                view_diagnostics=view_diags,
                config=cfg.black_litterman,
                p_matrix_qual=P_qual,
                sigma_matrix=sigma
            )
            
            # Black-Litterman Priors and Posterior
            market_caps = data_stores['market_caps'].loc[rebalance_date, final_universe]
            market_weights = market_caps / market_caps.sum()
            lambda_aversion = calculate_risk_aversion(data_stores['market_excess_returns'].loc[:rebalance_date])
            pi_prior = calculate_equilibrium_returns(lambda_aversion, sigma, market_weights)
            mu_posterior = calculate_posterior_returns(pi_prior, sigma, P, Q, Omega, cfg.black_litterman.tau)
            
            # Optimizer
            w_var = cp.Variable(len(final_universe))
            constraints = build_optimizer_constraints(w_var, cfg.optimizer, final_universe)
            target_weights_series = run_mean_variance_optimization(mu_posterior, sigma, lambda_aversion, constraints)
            if target_weights_series is None: target_weights = market_weights.copy()
            else: target_weights = target_weights_series
            
        except Exception as e:
            import traceback
            print(f"  -> ERROR in model pipeline: {e}. Holding previous positions.")
            traceback.print_exc()
            continue

        # --- C. Generate and Execute Trades ---
        target_shares_df = calculate_target_shares(target_weights, portfolio_value, current_prices)
        target_shares_df = resolve_fractional_shares(target_shares_df)
        trade_list = compute_trade_list(p_ledger.holdings.copy(), target_shares_df.set_index('ticker')['target_shares'])
        
        if not trade_list.empty:
            p_ledger.execute_trades(trade_list, current_prices, rebalance_date, cost_per_trade_bps=cfg.backtest.costs_bps)
            for _, trade in trade_list.iterrows():
                if trade['action'] == 'BUY':
                    tax_ledger.record_buy(trade['ticker'], trade['delta_shares'], current_prices[trade['ticker']], rebalance_date)
                elif trade['action'] == 'SELL':
                    tax_ledger.record_sell(trade['ticker'], abs(trade['delta_shares']), current_prices[trade['ticker']], rebalance_date)
        
        p_ledger.record_state(rebalance_date, current_prices)

    # --- 4. Post-Backtest Analysis ---
    results_history = p_ledger.get_history_df()
    if len(results_history) <= 1:
        print("\n--- No trades were executed. Cannot compute performance metrics. ---")
        return

    results_history['returns'] = results_history['portfolio_value'].pct_change().fillna(0)
    
    total_years = (results_history.index[-1] - results_history.index[0]).days / 365.25
    periods_per_year = len(rebalance_dates) / total_years if total_years > 0 else 0
    
    metrics = compute_performance_metrics(
        results_history['returns'],
        data_stores['market_simple_returns'].reindex(results_history.index).ffill(),
        data_stores['risk_free_daily']['rf_daily'].reindex(results_history.index).ffill(),
        periods_per_year=periods_per_year
    )
    print("\n--- PERFORMANCE SUMMARY ---\n", metrics)
    
    # Tax Calculation
    monthly_taxes = [
        calculate_monthly_taxes(tax_ledger.sales_log, date.year, date.month)
        for date in pd.date_range(cfg.backtest.start, cfg.backtest.end, freq='M')
    ]
    generate_darf_excel_report(monthly_taxes, output_path)

    # --- 5. Save All Artifacts ---
    results_history.to_csv(output_path / "portfolio_history.csv")
    metrics.to_json(output_path / "performance_summary.json", indent=4)
    pd.DataFrame(tax_ledger.sales_log).to_csv(output_path / "sales_log.csv")
    shutil.copy(config_path, output_path / "config.yaml")

    fig_equity = plot_equity_curve(results_history['returns'], data_stores['market_simple_returns'])
    fig_equity.savefig(output_path / "equity_curve.png", dpi=300)
    fig_drawdown = plot_drawdowns(results_history['returns'])
    fig_drawdown.savefig(output_path / "drawdowns.png", dpi=300)
    
    print(f"\nRun complete. All artifacts saved to:\n{output_path}")
    
if __name__ == '__main__':
    # This script is intended to be run from the command line, e.g.:
    # python scripts/run_backtest.py --config config/portfolio_A.yaml
    # A user must first create the config file and run update_data.py & build_factors.py
    main()