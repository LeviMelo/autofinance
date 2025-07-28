import pandas as pd
import numpy as np
from typing import Tuple, Optional, Any
from mvgarch.ugarch import UGARCH
from mvgarch.mgarch import DCCGARCH

# Define type hints for clarity
DccFitResult = Any # The specific type is complex
ForecastedCorrMatrix = Optional[np.ndarray]

def fit_dcc_model(
    standardized_residuals: pd.DataFrame
) -> Tuple[Optional[DccFitResult], ForecastedCorrMatrix]:
    """
    Fits a DCC(1,1)-GARCH(1,1) model using the mvgarch library.
    """
    if standardized_residuals.isnull().values.any():
        print("Warning: DCC input contains NaNs. Filling with 0 before fitting.")
        standardized_residuals = standardized_residuals.fillna(0)

    num_assets = standardized_residuals.shape[1]
    asset_names = standardized_residuals.columns.tolist()
    
    if num_assets < 2:
        print("Warning: DCC model requires at least 2 assets. Skipping.")
        return None, None
    
    try:
        ugarchs = [UGARCH(order=(1, 1)) for _ in range(num_assets)]
        
        dcc = DCCGARCH()
        dcc.spec(ugarch_objs=ugarchs, returns=standardized_residuals)
        
        # FIXED: Removed the unsupported 'disp' argument
        fit_result = dcc.fit()

        dcc.forecast(n_ahead=1)
        
        forecasted_corr_3d = dcc.fc_cor
        
        if forecasted_corr_3d is None or forecasted_corr_3d.shape != (num_assets, num_assets, 1):
            raise ValueError(f"Unexpected DCC forecast shape; expected {(num_assets, num_assets, 1)}")

        corr_matrix = forecasted_corr_3d[:, :, 0]
        
        if dcc.assets != asset_names:
            print("Warning: DCC model reordered assets. Re-indexing correlation matrix.")
            temp_df = pd.DataFrame(corr_matrix, index=dcc.assets, columns=dcc.assets)
            corr_matrix = temp_df.reindex(index=asset_names, columns=asset_names).values

        return fit_result, corr_matrix

    except Exception as e:
        print(f"Warning: mvgarch DCC model failed to fit. Error: {e}")
        return None, None


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