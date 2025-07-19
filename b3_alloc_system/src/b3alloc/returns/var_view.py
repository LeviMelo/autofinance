import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from typing import Tuple, Dict, Optional

from ..config import VarViewConfig

def create_var_view(
    log_returns_df: pd.DataFrame,
    config: VarViewConfig
) -> Tuple[Optional[pd.Series], Dict]:
    """
    Generates expected excess returns based on a Vector Autoregression (VAR) model.

    This function fits a VAR(p) model to the multivariate time series of
    log returns to produce a one-step-ahead forecast.

    Args:
        log_returns_df: A wide-format DataFrame of daily log excess returns for
                        all assets in the universe.
        config: The VarViewConfig object with model parameters.

    Returns:
        A tuple containing:
        - A Series of the final annualized expected excess returns (the "view").
        - A dictionary of model diagnostics (selected lag order, stability).
    """
    print("Generating return views from VAR model...")
    
    # VAR models are sensitive to NaNs. We must use a common, non-null history.
    data = log_returns_df.dropna()
    
    if data.shape[0] < 100 or data.shape[1] < 2:
        print("Warning: Not enough data or assets for a meaningful VAR model. Skipping.")
        return None, {"error": "Insufficient data"}
        
    diagnostics = {}
    
    # 1. Fit the VAR model
    # The statsmodels VAR implementation automatically selects the lag order if
    # maxlags is provided to the fit method.
    model = VAR(data)
    
    try:
        # The `fit` method can select the best lag order automatically
        results = model.fit(maxlags=config.max_lag, ic=config.criterion)
        
        diagnostics['selected_lag_order'] = results.k_ar
        diagnostics['is_stable'] = results.is_stable()
        
        if not diagnostics['is_stable']:
            print("Warning: Fitted VAR model is unstable. Forecasts may be unreliable.")
            # Proceeding anyway as per spec, but flagging it.
            
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        diagnostics['error'] = str(e)
        return None, diagnostics

    # 2. Generate one-step-ahead forecast
    # We need the last `p` observations to make the next forecast.
    p = results.k_ar
    last_observations = data.values[-p:]
    
    forecast_log_daily = results.forecast(y=last_observations, steps=1)
    
    # The forecast is a numpy array; convert it to a pandas Series
    forecast_log_daily_s = pd.Series(forecast_log_daily[0], index=data.columns)

    # 3. Convert forecast to the required format (annualized simple excess return)
    # The VAR model was fit on daily log returns.
    # Step 3a: Convert daily log return to daily simple return
    # simple_return = exp(log_return) - 1
    forecast_simple_daily = np.exp(forecast_log_daily_s) - 1
    
    # Step 3b: Annualize the daily simple return
    TRADING_DAYS_PER_YEAR = 252
    mu_view = forecast_simple_daily * TRADING_DAYS_PER_YEAR
    mu_view.name = "var_expected_returns"

    print(f"Successfully generated VAR view with lag order p={p}.")
    return mu_view, diagnostics


if __name__ == '__main__':
    from ..config import load_config
    from pathlib import Path
    import shutil

    # Create dummy config
    dummy_yaml = """
    return_engine:
      var:
        max_lag: 10
        criterion: bic
        log_returns: true
      factor: {lookback_days: 756, include_alpha: false, premium_estimator: long_term_mean}
    # Add other sections to satisfy pydantic
    data: {start: '', end: '', tickers_file: '', selic_series: 0, publish_lag_days: 0}
    risk_engine: {garch: {dist: gaussian, min_obs: 50, refit_freq_days: 21}, dcc: {a_init: 0.02, b_init: 0.97}, shrinkage: {method: ledoit_wolf, floor: 0.0}}
    black_litterman: {tau: 0.05, confidence: {method: rmse_based, factor_scaler: 1.0, var_scaler: 1.0}}
    optimizer: {objective: max_sharpe, long_only: true, name_cap: 0.1, sector_cap: 0.25, turnover_penalty_bps: 0}
    backtest: {lookback_years: 1, rebalance: monthly, start: '', end: '', costs_bps: 0}
    """
    temp_dir = Path("./temp_var_config"); temp_dir.mkdir(exist_ok=True)
    dummy_config_path = temp_dir / "var_config.yaml"
    dummy_config_path.write_text(dummy_yaml)
    config = load_config(dummy_config_path)

    # Generate synthetic data exhibiting mean reversion
    np.random.seed(42)
    n_obs, n_assets = 500, 2
    dates = pd.date_range("2020-01-01", periods=n_obs)
    tickers = ['MEAN_REVERTER', 'MOMENTUM_STOCK']
    
    returns = pd.DataFrame(np.random.randn(n_obs, n_assets) * 0.01, index=dates, columns=tickers)
    
    # Create mean-reverting behavior in the first asset
    for t in range(1, n_obs):
        returns.iloc[t, 0] = -0.3 * returns.iloc[t-1, 0] + 0.1 * returns.iloc[t-1, 1] + np.random.randn() * 0.01
    
    # Add a recent negative shock to the mean-reverting asset
    returns.iloc[-1, 0] = -0.05
    
    print("\n--- Last observation (input to forecast) ---")
    print(returns.tail(1))
    
    try:
        mu_view, diags = create_var_view(returns, config.return_engine.var)
        
        print("\n--- Diagnostics ---")
        if diags.get("error") is None:
            print(diags)
            
            print("\n--- Final Expected Returns View (Annualized) ---")
            print(mu_view)
            
            # Validation
            assert diags['selected_lag_order'] > 0
            
            # Because the last return of MEAN_REVERTER was very negative, a mean-reverting
            # model should predict a positive return for the next period.
            assert mu_view['MEAN_REVERTER'] > 0, "VAR should predict a positive return after a negative shock for a mean-reverting asset."
            print("\nOK: VAR forecast is consistent with mean-reverting dynamics.")
        else:
            print(f"Test failed with error: {diags['error']}")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)