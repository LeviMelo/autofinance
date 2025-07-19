import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Dict, Tuple

from ..config import Config
from ..utils_dates import generate_rebalance_dates
from ..preprocess.clean import get_liquid_universe
from ..risk.risk_engine import build_covariance_matrix
from ..returns.ff_view import create_fama_french_view
from ..returns.var_view import create_var_view
from ..bl.black_litterman import calculate_risk_aversion, calculate_equilibrium_returns, calculate_posterior_returns
from ..bl.view_builder import build_absolute_views
from ..bl.confidence import estimate_view_uncertainty
from ..optimize.mean_variance import run_mean_variance_optimization
from ..optimize.constraints import build_optimizer_constraints
from ..trades.trade_calculator import calculate_target_shares, resolve_fractional_shares, compute_trade_list
from .portfolio_accounting import PortfolioLedger

class BacktestEngine:
    """
    Orchestrates the entire rolling-window backtesting simulation using a
    detailed portfolio ledger for realistic accounting.
    """
    def __init__(self, config: Config, data_stores: Dict[str, pd.DataFrame]):
        self.config = config
        self.data_stores = data_stores
        self.ledger = PortfolioLedger(initial_capital=config.backtest.get('initial_capital', 1_000_000.0))

    def run_backtest(self) -> pd.DataFrame:
        """
        Executes the main backtesting loop.
        """
        cfg_bt = self.config.backtest
        rebalance_dates = generate_rebalance_dates(cfg_bt['start'], cfg_bt['end'], cfg_bt['rebalance'])
        prices_wide = self.data_stores['prices'].pivot(index='date', columns='ticker', values='adj_close')

        print(f"Starting backtest from {rebalance_dates[0].date()} to {rebalance_dates[-1].date()}...")

        for rebalance_date in rebalance_dates:
            print(f"\n--- Processing Rebalance Date: {rebalance_date.date()} ---")
            
            # --- State at Start of Period ---
            current_prices = prices_wide.loc[rebalance_date]
            portfolio_value_start = self.ledger.get_portfolio_value(current_prices, rebalance_date)
            
            # --- Run Model Pipeline to Get Target Weights ---
            target_weights, diags = self._run_rebalance_pipeline(rebalance_date, current_prices)
            
            if target_weights is None:
                print("  -> Rebalance pipeline failed. No trades will be executed.")
                continue # Skip to next rebalance date, holding current positions
            
            # --- Generate and Execute Trades ---
            # 1. Convert weights to target share counts
            target_shares_df = calculate_target_shares(target_weights, portfolio_value_start, current_prices)
            target_shares_df = resolve_fractional_shares(target_shares_df)
            
            # 2. Compute trade list
            current_holdings = self.ledger.holdings.copy()
            target_holdings = target_shares_df.set_index('ticker')['target_shares']
            trade_list = compute_trade_list(current_holdings, target_holdings)
            
            # 3. Execute trades in the ledger
            if not trade_list.empty:
                print(f"  -> Executing {len(trade_list)} trades...")
                self.ledger.execute_trades(
                    trade_list,
                    current_prices,
                    rebalance_date,
                    cost_per_trade_bps=cfg_bt['costs_bps']
                )
            else:
                print("  -> No trades needed to reach target allocation.")

            # Record state at the end of the rebalance day
            self.ledger.record_state(rebalance_date, current_prices)

        print("\nBacktest finished.")
        return self.ledger.get_history_df()

    def _run_rebalance_pipeline(self, rebalance_date: pd.Timestamp, current_prices: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Executes the full data prep, risk, return, and optimization pipeline.
        """
        cfg = self.config
        lookback_start = rebalance_date - pd.DateOffset(years=cfg.backtest['lookback_years'])
        
        # --- Data Slicing and Universe ---
        history_prices = self.data_stores['prices'][
            (self.data_stores['prices']['date'] >= lookback_start) &
            (self.data_stores['prices']['date'] <= rebalance_date)
        ]
        prices_wide = history_prices.pivot(index='date', columns='ticker', values='adj_close')
        volume_wide = history_prices.pivot(index='date', columns='ticker', values='volume')
        
        liquid_universe = get_liquid_universe(
            prices_wide, volume_wide, rebalance_date, lookback_days=365
        )
        if len(liquid_universe) < 2:
            return None, {"error": "Insufficient liquid assets"}
            
        # --- Model Execution ---
        try:
            log_returns = self.data_stores['log_returns'].loc[lookback_start:rebalance_date, liquid_universe]
            
            sigma_final, diags = build_covariance_matrix(log_returns.dropna(), cfg.risk_engine)
            final_universe = sorted(sigma_final.columns.tolist())
            
            # --- View Generation ---
            ff_betas, ff_view = create_fama_french_view(
                self.data_stores['excess_log_returns'].loc[lookback_start:rebalance_date, final_universe],
                self.data_stores['factors'].loc[lookback_start:rebalance_date],
                cfg.return_engine.factor
            )
            var_view, var_diags = create_var_view(log_returns.dropna(), cfg.return_engine.var)
            views = {'ff_view': ff_view, 'var_view': var_view}
            
            market_caps = self.data_stores['market_caps'].loc[rebalance_date, final_universe]
            market_weights = market_caps / market_caps.sum()
            lambda_aversion = calculate_risk_aversion(self.data_stores['market_excess_returns'].loc[:rebalance_date])
            pi_prior = calculate_equilibrium_returns(lambda_aversion, sigma_final, market_weights)
            
            P, Q = build_absolute_views(views, final_universe)
            
            if P is None or Q is None:
                print("  -> No valid views generated. Falling back to prior.")
                mu_posterior = pi_prior
            else:
                Omega = estimate_view_uncertainty(
                    views,
                    {'ff_view': {'res_var': ff_betas['residual_variance']}, 'var_view': var_diags},
                    cfg.black_litterman
                )
                mu_posterior = calculate_posterior_returns(pi_prior, sigma_final, P, Q, Omega, cfg.black_litterman['tau'])

            # --- Optimization ---
            w_var = cp.Variable(len(final_universe))
            constraints = build_optimizer_constraints(w_var, cfg.optimizer, final_universe)
            optimal_weights = run_mean_variance_optimization(mu_posterior, sigma_final, lambda_aversion, constraints)
            
            if optimal_weights is None:
                print("  -> Optimizer failed. Falling back to market weights.")
                optimal_weights = market_weights.copy()
            
            return optimal_weights, diags

        except Exception as e:
            import traceback
            print(f"  -> ERROR during rebalance pipeline: {e}")
            traceback.print_exc()
            return None, {"error": str(e)}

if __name__ == '__main__':
    print("--- Backtest Engine (Ledger-Based) ---")
    print("This module is designed to be driven by a master script like 'run_backtest.py'.")
    print("Its standalone test would be complex, requiring the instantiation of a full")
    print("Config object and a complete 'data_stores' dictionary.")
    print("The individual component modules have their own unit tests.")