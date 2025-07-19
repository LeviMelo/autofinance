import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, List, Optional
from joblib import Parallel, delayed

from ..config import FactorViewConfig

def estimate_factor_betas(
    asset_excess_returns: pd.Series,
    base_factor_returns: pd.DataFrame,
    fx_factor_return: Optional[pd.Series] = None
) -> Tuple[Optional[pd.Series], Optional[Dict]]:
    """
    Estimates factor betas for a single asset via OLS regression.

    Dynamically includes an FX factor in the regression if it is provided.

    Args:
        asset_excess_returns: A Series of daily excess returns for one asset.
        base_factor_returns: A DataFrame with daily returns for Mkt_excess, SMB, and HML.
        fx_factor_return: An optional Series of daily FX factor returns.

    Returns:
        A tuple containing:
        - A Series of estimated betas (including alpha and potentially fx_beta).
        - A dictionary of regression diagnostics (R-squared, p-values).
    """
    # Combine all potential regressors
    all_factors = [base_factor_returns]
    if fx_factor_return is not None:
        all_factors.append(fx_factor_return)
        
    # Align all data and drop any days with missing values
    data = pd.concat([asset_excess_returns] + all_factors, axis=1).dropna()
    
    if data.shape[0] < 60: # Not enough data for a meaningful regression
        return None, None

    Y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    X = sm.add_constant(X) # Add a constant to estimate alpha

    model = sm.OLS(Y, X)
    results = model.fit()
    
    betas = results.params
    betas = betas.rename({'const': 'alpha'})
    
    diagnostics = {
        'r_squared': results.rsquared_adj,
        'p_values': results.pvalues,
        'residual_variance': results.resid.var()
    }
    
    return betas, diagnostics


def estimate_factor_premia(
    factor_returns: pd.DataFrame,
    estimator: str = "long_term_mean"
) -> pd.Series:
    """
    Estimates the forward-looking factor premia.
    Now handles an arbitrary number of factors.
    """
    TRADING_DAYS_PER_YEAR = 252
    
    if estimator == "long_term_mean":
        daily_premia = factor_returns.mean()
        annual_premia = daily_premia * TRADING_DAYS_PER_YEAR
        return annual_premia
    else:
        raise NotImplementedError(f"Estimator '{estimator}' not implemented.")


def create_fama_french_view(
    asset_returns_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
    config: FactorViewConfig,
    fx_returns_df: Optional[pd.DataFrame] = None,
    asset_fx_sensitivity: Optional[Dict[str, bool]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates expected excess returns based on a dynamic factor model.
    """
    print("Generating return views from factor model (FX-aware)...")
    n_jobs = -1
    asset_fx_sensitivity = asset_fx_sensitivity or {}

    # --- 1. Estimate betas for all assets in parallel ---
    print(f"Estimating factor betas for {asset_returns_df.shape[1]} assets...")
    beta_results = Parallel(n_jobs=n_jobs)(
        delayed(estimate_factor_betas)(
            asset_returns_df[ticker],
            factor_returns_df,
            fx_returns_df.iloc[:, 0] if fx_returns_df is not None and asset_fx_sensitivity.get(ticker, False) else None
        )
        for ticker in asset_returns_df.columns
    )
    
    all_betas = {}
    for ticker, (betas, diags) in zip(asset_returns_df.columns, beta_results):
        if betas is not None:
            all_betas[ticker] = betas
            
    if not all_betas:
        raise RuntimeError("Failed to estimate betas for any asset.")
        
    betas_df = pd.DataFrame(all_betas).T.fillna(0) # Fill missing betas (e.g., FX beta for non-sensitive assets) with 0

    # --- 2. Estimate historical factor premia ---
    print("Estimating historical factor premia (including FX)...")
    # Combine all possible factors for premia calculation
    full_factor_panel = factor_returns_df.copy()
    if fx_returns_df is not None:
        full_factor_panel = pd.concat([full_factor_panel, fx_returns_df], axis=1)

    factor_premia = estimate_factor_premia(
        full_factor_panel.dropna(), config.premium_estimator
    )
    
    # --- 3. Project expected returns ---
    print("Projecting expected returns from betas and premia...")
    factor_betas = betas_df.drop(columns=['alpha'], errors='ignore')
    
    # Align factor betas and premia columns before dot product
    aligned_betas, aligned_premia = factor_betas.align(factor_premia, join='left', axis=1)
    aligned_betas = aligned_betas.fillna(0) # Ensure any non-estimated betas are 0
    aligned_premia = aligned_premia.fillna(0)
    
    expected_returns = aligned_betas.dot(aligned_premia)
    
    if config.include_alpha and 'alpha' in betas_df.columns:
        expected_returns += betas_df['alpha']
    
    expected_returns.name = "ff_expected_returns"
    
    print("Successfully generated FX-aware Fama-French return view.")
    return betas_df, expected_returns


if __name__ == '__main__':
    from ..config import load_config
    from pathlib import Path
    import shutil

    # Create dummy config
    # ... (same as before, FactorViewConfig is sufficient) ...
    # (assuming previous test setup code for config)

    # --- GIVEN ---
    np.random.seed(42)
    n_obs, n_assets = 1000, 2
    dates = pd.date_range("2018-01-01", periods=n_obs)
    tickers = ['PETR4.SA', 'AAPL34.SA'] # A domestic stock and a BDR
    
    factors = pd.DataFrame(np.random.randn(n_obs, 3) * 0.01, index=dates, columns=['mkt_excess', 'smb', 'hml'])
    fx_factor = pd.DataFrame(np.random.randn(n_obs, 1) * 0.005, index=dates, columns=['USD_BRL_log_return'])
    
    # Create asset returns with specific sensitivities
    asset_returns = pd.DataFrame(index=dates, columns=tickers)
    asset_returns['PETR4.SA'] = 1.2 * factors['mkt_excess'] + np.random.randn(n_obs) * 0.015 # No FX beta
    asset_returns['AAPL34.SA'] = 0.8 * factors['mkt_excess'] + 1.0 * fx_factor['USD_BRL_log_return'] + np.random.randn(n_obs) * 0.01 # High FX beta
    
    # Define which assets are sensitive
    fx_sensitivity_map = {'AAPL34.SA': True}
    
    # Dummy config setup
    dummy_yaml = """
    return_engine:
      factor:
        lookback_days: 756
        include_alpha: false
        premium_estimator: long_term_mean
      var: {max_lag: 1, criterion: bic, log_returns: true}
    data: {start: '', end: '', tickers_file: '', selic_series: 0, publish_lag_days: 0}
    risk_engine: {garch: {dist: gaussian, min_obs: 50, refit_freq_days: 21}, dcc: {a_init: 0.02, b_init: 0.97}, shrinkage: {method: ledoit_wolf, floor: 0.0}}
    black_litterman: {tau: 0.05, confidence: {method: rmse_based, factor_scaler: 1.0, var_scaler: 1.0}}
    optimizer: {objective: max_sharpe, long_only: true, name_cap: 0.1, sector_cap: 0.25, turnover_penalty_bps: 0}
    backtest: {lookback_years: 1, rebalance: monthly, start: '', end: '', costs_bps: 0}
    """
    temp_dir = Path("./temp_ff_view_config"); temp_dir.mkdir(exist_ok=True)
    dummy_config_path = temp_dir / "config.yaml"
    dummy_config_path.write_text(dummy_yaml)
    config = load_config(dummy_config_path)


    # --- WHEN ---
    betas, mu_view = create_fama_french_view(
        asset_returns,
        factors,
        config.return_engine.factor,
        fx_returns_df=fx_factor,
        asset_fx_sensitivity=fx_sensitivity_map
    )
    
    # --- THEN ---
    print("\n--- Estimated Betas (including FX) ---")
    print(betas)
    
    print("\n--- Final Expected Returns View (Annualized) ---")
    print(mu_view)
    
    # --- Validation ---
    # 1. Betas for the BDR should include a non-zero FX beta
    fx_beta_col_name = fx_factor.columns[0]
    assert fx_beta_col_name in betas.columns
    assert betas.loc['AAPL34.SA', fx_beta_col_name] > 0.8 # Should be close to 1.0
    
    # 2. Betas for the domestic stock should have a zero FX beta
    assert np.isclose(betas.loc['PETR4.SA', fx_beta_col_name], 0.0)
    
    print("\nOK: Model correctly estimates FX beta for sensitive assets and zero for others.")
    
    # 3. Check mu_view calculation
    # Expected return for AAPL34.SA should be influenced by the FX premium
    combined_factors = pd.concat([factors, fx_factor], axis=1)
    premia = estimate_factor_premia(combined_factors)
    
    expected_mu_aapl = (betas.loc['AAPL34.SA'][premia.index] * premia).sum()
    assert np.isclose(mu_view['AAPL34.SA'], expected_mu_aapl)
    print("OK: Expected return calculation correctly incorporates the FX premium.")
    
    # Cleanup
    shutil.rmtree(temp_dir)