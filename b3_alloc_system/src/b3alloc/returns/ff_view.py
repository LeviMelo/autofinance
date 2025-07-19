import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, List
from joblib import Parallel, delayed

from ..config import FactorViewConfig

def estimate_factor_betas(
    asset_excess_returns: pd.Series,
    factor_returns: pd.DataFrame
) -> Tuple[pd.Series, Dict]:
    """
    Estimates the Fama-French 3-factor betas for a single asset via OLS regression.

    Args:
        asset_excess_returns: A Series of daily excess returns for one asset.
        factor_returns: A DataFrame with daily returns for Mkt_excess, SMB, and HML.

    Returns:
        A tuple containing:
        - A Series of estimated betas (including alpha, the intercept).
        - A dictionary of regression diagnostics (R-squared, p-values).
    """
    # Align data and drop any days with missing values
    data = pd.concat([asset_excess_returns, factor_returns], axis=1).dropna()
    
    if data.shape[0] < 60: # Not enough data for a meaningful regression
        return None, None

    Y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    X = sm.add_constant(X) # Add a constant to estimate alpha

    model = sm.OLS(Y, X)
    results = model.fit()
    
    betas = results.params
    betas.rename({'const': 'alpha'}, inplace=True)
    
    diagnostics = {
        'r_squared': results.rsquared_adj,
        'p_values': results.pvalues
    }
    
    return betas, diagnostics


def estimate_factor_premia(
    factor_returns: pd.DataFrame,
    estimator: str = "long_term_mean"
) -> pd.Series:
    """
    Estimates the forward-looking factor premia.

    Args:
        factor_returns: A DataFrame of historical daily factor returns.
        estimator: The method to use ('long_term_mean', etc.).

    Returns:
        A Series containing the estimated annual premium for each factor.
    """
    # As per spec, use arithmetic mean and annualize.
    # Trading days per year constant is typically 252.
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
    config: FactorViewConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates expected excess returns based on the Fama-French 3-factor model.

    Args:
        asset_returns_df: Wide-format DataFrame of daily excess returns for all assets.
        factor_returns_df: DataFrame of daily FF3 factor returns.
        config: The FactorViewConfig object.

    Returns:
        A tuple containing:
        - A DataFrame of the estimated betas for all assets.
        - A Series of the final annualized expected excess returns (the "view").
    """
    print("Generating return views from Fama-French 3-factor model...")
    n_jobs = -1

    # 1. Estimate betas for all assets in parallel
    print(f"Estimating factor betas for {asset_returns_df.shape[1]} assets...")
    beta_results = Parallel(n_jobs=n_jobs)(
        delayed(estimate_factor_betas)(asset_returns_df[ticker], factor_returns_df)
        for ticker in asset_returns_df.columns
    )
    
    # Process results, filtering out failures
    all_betas = {}
    successful_tickers = []
    for ticker, (betas, diags) in zip(asset_returns_df.columns, beta_results):
        if betas is not None:
            all_betas[ticker] = betas
            successful_tickers.append(ticker)

    if not all_betas:
        raise RuntimeError("Failed to estimate betas for any asset.")
        
    betas_df = pd.DataFrame(all_betas).T # Transpose to get tickers as rows

    # 2. Estimate historical factor premia
    print("Estimating historical factor premia...")
    factor_premia = estimate_factor_premia(
        factor_returns_df, config.premium_estimator
    )
    
    # 3. Project expected returns
    print("Projecting expected returns from betas and premia...")
    # Get the factor betas (excluding alpha)
    factor_betas = betas_df.drop(columns=['alpha'])
    
    # Project returns: mu = Beta * Premia
    # This matrix multiplication calculates the sum of (beta * premium) for each asset
    expected_returns = factor_betas.dot(factor_premia)
    
    # Optionally, add back a shrunk version of the estimated alpha
    if config.include_alpha:
        # Simple alpha inclusion for now. A more advanced version would shrink it.
        expected_returns += betas_df['alpha']
    
    expected_returns.name = "ff_expected_returns"
    
    print("Successfully generated Fama-French return view.")
    return betas_df, expected_returns


if __name__ == '__main__':
    from ..config import load_config
    from pathlib import Path
    import shutil

    # Create dummy config
    dummy_yaml = """
    return_engine:
      factor:
        lookback_days: 756
        include_alpha: false
        premium_estimator: long_term_mean
      var: {max_lag: 1, criterion: bic, log_returns: true}
    # Add other sections to satisfy pydantic
    data: {start: '', end: '', tickers_file: '', selic_series: 0, publish_lag_days: 0}
    risk_engine: {garch: {dist: gaussian, min_obs: 50, refit_freq_days: 21}, dcc: {a_init: 0.02, b_init: 0.97}, shrinkage: {method: ledoit_wolf, floor: 0.0}}
    black_litterman: {tau: 0.05, confidence: {method: rmse_based, factor_scaler: 1.0, var_scaler: 1.0}}
    optimizer: {objective: max_sharpe, long_only: true, name_cap: 0.1, sector_cap: 0.25, turnover_penalty_bps: 0}
    backtest: {lookback_years: 1, rebalance: monthly, start: '', end: '', costs_bps: 0}
    """
    temp_dir = Path("./temp_ff_config"); temp_dir.mkdir(exist_ok=True)
    dummy_config_path = temp_dir / "ff_config.yaml"
    dummy_config_path.write_text(dummy_yaml)
    config = load_config(dummy_config_path)

    # Generate synthetic data
    np.random.seed(42)
    n_obs, n_assets = 1000, 4
    dates = pd.date_range("2018-01-01", periods=n_obs)
    tickers = ['HIGH_BETA', 'LOW_BETA', 'SIZE_SENSITIVE', 'VALUE_SENSITIVE']
    
    factors = pd.DataFrame(np.random.randn(n_obs, 3) * 0.01, index=dates, columns=['mkt_excess', 'smb', 'hml'])
    factors['mkt_excess'] += 0.0003 # Add a positive drift
    
    # Create asset returns with specific sensitivities
    asset_returns = pd.DataFrame(index=dates, columns=tickers)
    asset_returns['HIGH_BETA'] = 1.5 * factors['mkt_excess'] + np.random.randn(n_obs) * 0.01
    asset_returns['LOW_BETA'] = 0.5 * factors['mkt_excess'] + np.random.randn(n_obs) * 0.01
    asset_returns['SIZE_SENSITIVE'] = 0.8 * factors['mkt_excess'] + 0.9 * factors['smb'] + np.random.randn(n_obs) * 0.01
    asset_returns['VALUE_SENSITIVE'] = 1.1 * factors['mkt_excess'] - 0.7 * factors['hml'] + np.random.randn(n_obs) * 0.01
    
    try:
        betas, mu_view = create_fama_french_view(asset_returns, factors, config.return_engine.factor)
        
        print("\n--- Estimated Betas ---")
        print(betas)
        
        print("\n--- Final Expected Returns View (Annualized) ---")
        print(mu_view)
        
        # Validation
        # HIGH_BETA should have the highest market beta
        assert betas.loc['HIGH_BETA', 'mkt_excess'] > betas.loc['LOW_BETA', 'mkt_excess']
        # HIGH_BETA should have higher expected return than LOW_BETA if mkt premium is positive
        assert mu_view['HIGH_BETA'] > mu_view['LOW_BETA']
        # SIZE_SENSITIVE should have a large SMB beta
        assert betas.loc['SIZE_SENSITIVE', 'smb'] > 0.8
        
        print("\nOK: View generation logic is consistent with factor sensitivities.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)