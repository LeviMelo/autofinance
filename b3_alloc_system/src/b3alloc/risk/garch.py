import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
from arch import arch_model
from arch.univariate import GARCH, StudentsT, Normal

# Define type hints for clarity
FitResult = Any # The specific type is complex, 'any' is sufficient here
ForecastResult = Optional[float]

def fit_garch_model(
    returns_series: pd.Series,
    dist: str = 'gaussian'
) -> Tuple[Optional[FitResult], ForecastResult]:
    """
    Fits a GARCH(1,1) model to a single asset's return series.

    Args:
        returns_series: A pandas Series of demeaned log excess returns for one asset.
        dist: The distribution to use for the innovations ('gaussian' or 'studentst').

    Returns:
        A tuple containing:
        - The fitted model result object (or None if convergence fails).
        - The one-step-ahead forecasted variance (or None if convergence fails).
    """
    if returns_series.isnull().any() or returns_series.empty:
        # print(f"Warning: GARCH input for {returns_series.name} contains NaNs or is empty. Skipping.")
        return None, None

    # GARCH models work best on demeaned series. We assume returns are already excess returns.
    # The model automatically estimates a mean, so we don't need to subtract it manually.
    # Rescaling by 100 can help with optimizer convergence.
    scaled_returns = returns_series * 100

    dist_name = dist.lower()

    # Define the GARCH(1,1) model
    model = arch_model(
        scaled_returns,
        vol='Garch',
        p=1,
        o=0,
        q=1,
        dist=dist_name
    )
    
    try:
        # Fit the model. disp='off' suppresses the output.
        fit_result = model.fit(update_freq=0, disp='off')
        
        # Check for successful convergence
        if not fit_result.convergence_flag == 0:
            # print(f"Warning: GARCH for {returns_series.name} did not converge. Status: {fit_result.convergence_flag}")
            return None, None
            
    except Exception as e:
        # print(f"Warning: GARCH for {returns_series.name} failed to fit. Error: {e}")
        return None, None

    # Forecast one step ahead
    forecast = fit_result.forecast(horizon=1, reindex=False)
    
    # The result is in a DataFrame. We need to extract the variance value.
    # The column name is 'h.1' for one-step-ahead variance.
    # Remember to scale it back down since we scaled the inputs by 100.
    forecasted_variance = forecast.variance.iloc[0, 0] / (100**2)
    
    return fit_result, forecasted_variance


if __name__ == '__main__':
    print("--- Running GARCH Module Standalone Test ---")

    # 1. Generate a synthetic return series with volatility clustering
    np.random.seed(42)
    n_obs = 1000
    dates = pd.date_range("2010-01-01", periods=n_obs)
    
    # Simulate a GARCH(1,1) process
    omega = 0.01
    alpha = 0.1
    beta = 0.88
    
    true_vol = np.zeros(n_obs)
    returns = np.zeros(n_obs)
    true_vol[0] = np.sqrt(omega / (1 - alpha - beta))
    returns[0] = np.random.normal(0, true_vol[0])
    
    for t in range(1, n_obs):
        true_vol[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * true_vol[t-1]**2)
        returns[t] = np.random.normal(0, true_vol[t])
        
    returns_series = pd.Series(returns, index=dates, name="SYNTHETIC_GARCH")

    print("\n--- Testing with synthetic GARCH data (Gaussian) ---")
    try:
        garch_fit, variance_forecast = fit_garch_model(returns_series, dist='gaussian')

        assert garch_fit is not None, "Model failed to fit on good data."
        assert variance_forecast is not None, "Forecast should not be None."
        
        print("Model fitted successfully.")
        print(garch_fit.summary())
        print(f"\nOne-step-ahead variance forecast: {variance_forecast:.8f}")
        
        # Check if estimated params are reasonable
        est_alpha = garch_fit.params['alpha[1]']
        est_beta = garch_fit.params['beta[1]']
        assert abs(est_alpha - alpha) < 0.05, f"Alpha estimate {est_alpha} is too far from true value {alpha}."
        assert abs(est_beta - beta) < 0.05, f"Beta estimate {est_beta} is too far from true value {beta}."
        print("\nOK: Estimated parameters are close to true simulation parameters.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()

    print("\n--- Testing failure case (constant series) ---")
    constant_series = pd.Series([0.001] * 200, name="CONSTANT")
    garch_fit_fail, var_fail = fit_garch_model(constant_series)
    
    assert garch_fit_fail is None, "Model should fail on constant series."
    assert var_fail is None, "Forecast should be None on failure."
    print("OK: Model correctly failed to converge on a constant series.")