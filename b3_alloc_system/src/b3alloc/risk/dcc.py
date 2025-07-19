import pandas as pd
import numpy as np
from typing import Tuple, Optional
from arch import arch_model

# Define type hints for clarity
DccFitResult = any # The specific type is complex
ForecastedCorrMatrix = Optional[np.ndarray]

def fit_dcc_model(
    standardized_residuals: pd.DataFrame
) -> Tuple[Optional[DccFitResult], ForecastedCorrMatrix]:
    """
    Fits a DCC(1,1)-GARCH(1,1) model to a set of standardized residuals.

    This implements the second stage of the DCC-GARCH process. By feeding the
    model pre-standardized residuals, we focus the estimation on the DCC
    parameters that govern the time-varying correlations.

    Args:
        standardized_residuals: A DataFrame where each column is an asset's
                                time series of standardized residuals (returns
                                divided by conditional volatility from GARCH).
                                The index is the date.

    Returns:
        A tuple containing:
        - The fitted DCC model result object (or None if convergence fails).
        - The one-step-ahead forecasted NxN correlation matrix (or None if fails).
    """
    if standardized_residuals.isnull().values.any():
        print("Warning: DCC input contains NaNs. Filling with 0 before fitting.")
        standardized_residuals = standardized_residuals.fillna(0)

    num_assets = standardized_residuals.shape[1]
    if num_assets < 2:
        print("Warning: DCC model requires at least 2 assets. Skipping.")
        return None, None
    
    # Define the GARCH specification for the DCC model.
    # The 'arch' library is designed to handle multiple series by passing
    # the entire DataFrame of residuals to the model constructor.
    # The original loop-based construction was incorrect.
    vol_model = arch_model(
        standardized_residuals, 
        vol='Garch', 
        p=1, q=1, 
        cov_type='DCC', 
        cov_p=1, cov_q=1
    )

    try:
        # Fit the model. Scaling is not needed as residuals are already standardized.
        # Options can be tuned for better convergence if needed.
        fit_result = vol_model.fit(disp='off', options={'maxiter': 500})
        
        if not fit_result.convergence_flag == 0:
            print(f"Warning: DCC model did not converge. Status: {fit_result.convergence_flag}")
            return None, None

    except Exception as e:
        print(f"Warning: DCC model failed to fit. Error: {e}")
        return None, None
        
    # Forecast one step ahead
    forecast = fit_result.forecast(horizon=1, reindex=False)
    
    # The forecasted correlation matrix is nested in the output.
    # The structure is forecast.correlation['h.1'][date_index]
    # We want the NxN matrix for the single forecasted date.
    forecast_date = forecast.correlation.index[0]
    corr_matrix = forecast.correlation.loc[forecast_date].values
    
    return fit_result, corr_matrix


if __name__ == '__main__':
    print("--- Running DCC Module Standalone Test ---")

    # 1. Generate synthetic standardized residuals with a changing correlation structure
    np.random.seed(42)
    n_obs = 1500
    dates = pd.date_range("2015-01-01", periods=n_obs)
    
    # Regime 1: High positive correlation (first 750 days)
    corr1 = np.array([[1.0, 0.8], [0.8, 1.0]])
    # Regime 2: Negative correlation (last 750 days)
    corr2 = np.array([[1.0, -0.6], [-0.6, 1.0]])
    
    chol1 = np.linalg.cholesky(corr1)
    chol2 = np.linalg.cholesky(corr2)
    
    # Independent random variables
    innovations = np.random.normal(0, 1, size=(n_obs, 2))
    
    # Create correlated residuals
    residuals1 = (chol1 @ innovations[:750].T).T
    residuals2 = (chol2 @ innovations[750:].T).T
    
    residuals = np.vstack([residuals1, residuals2])
    residuals_df = pd.DataFrame(residuals, index=dates, columns=['ASSET_X', 'ASSET_Y'])

    print("\n--- Testing with synthetic data ---")
    print(f"True correlation in first half: {corr1[0,1]}")
    print(f"True correlation in second half: {corr2[0,1]}")
    
    try:
        dcc_fit, forecasted_corr = fit_dcc_model(residuals_df)
        
        assert dcc_fit is not None, "DCC model failed to fit on good data."
        assert forecasted_corr is not None, "DCC forecast should not be None."
        
        print("\nDCC model fitted successfully.")
        
        print("\nForecasted Correlation Matrix (t+1):")
        print(forecasted_corr)
        
        # Validation
        # The forecasted correlation should be much closer to the second regime's
        # correlation (-0.6) than the first's (0.8).
        forecasted_xy_corr = forecasted_corr[0, 1]
        
        print(f"\nForecasted correlation between X and Y: {forecasted_xy_corr:.4f}")
        assert forecasted_xy_corr < 0, "Forecasted correlation should be negative."
        assert abs(forecasted_xy_corr - (-0.6)) < 0.2, "Forecasted correlation is not close to the recent regime."
        print("OK: Forecasted correlation correctly reflects the most recent correlation regime.")

        # Check matrix properties
        assert forecasted_corr.shape == (2, 2), "Matrix shape is incorrect."
        assert np.allclose(np.diag(forecasted_corr), [1.0, 1.0]), "Diagonal must be all ones."
        assert np.allclose(forecasted_corr, forecasted_corr.T), "Matrix must be symmetric."
        print("OK: Forecasted matrix has valid properties.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()