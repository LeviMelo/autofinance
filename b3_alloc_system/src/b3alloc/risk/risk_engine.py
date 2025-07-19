import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed

from .garch import fit_garch_model
from .dcc import fit_dcc_model
from .shrinkage import apply_ledoit_wolf_shrinkage
from ..config import RiskEngineConfig

def build_covariance_matrix(
    returns_df: pd.DataFrame,
    config: RiskEngineConfig,
    fx_returns_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Orchestrates the construction of the forward-looking covariance matrix, now
    with optional FX factor integration.

    If an FX return series is provided, it is included in the GARCH-DCC
    estimation to capture its volatility and correlations with other assets.
    It is then removed before the final matrix is returned to the optimizer.

    Args:
        returns_df: A wide-format DataFrame of log excess returns for the assets.
        config: The RiskEngineConfig object.
        fx_returns_df: An optional single-column DataFrame of FX log returns.

    Returns:
        A tuple containing:
        - A pandas DataFrame representing the final, shrunk NxN asset covariance matrix.
        - A dictionary containing diagnostic information.
    """
    print("Building dynamic covariance matrix (FX-aware)...")
    if returns_df.isnull().all().any():
        raise ValueError("A column in the returns_df is all NaN. Check input data.")

    # --- 1. Combine Asset and FX Returns ---
    model_input_returns = returns_df.copy()
    fx_col_name = None
    if fx_returns_df is not None and not fx_returns_df.empty:
        # Align and combine
        fx_col_name = fx_returns_df.columns[0]
        model_input_returns = pd.concat([returns_df, fx_returns_df], axis=1).dropna(axis=0, how='any')
        print(f"  -> Including '{fx_col_name}' in risk estimation.")

    n_jobs = -1
    
    # --- 2. Univariate GARCH Fitting (Parallelized) ---
    print(f"Fitting GARCH(1,1) for {model_input_returns.shape[1]} series (assets + FX)...")
    garch_results = Parallel(n_jobs=n_jobs)(
        delayed(fit_garch_model)(model_input_returns[ticker].dropna(), config.garch.dist)
        for ticker in model_input_returns.columns
    )
    
    # --- 3. Process GARCH results ---
    successful_series = []
    standardized_residuals = {}
    variance_forecasts = {}

    for series_name, (fit_result, var_forecast) in zip(model_input_returns.columns, garch_results):
        if fit_result and var_forecast:
            successful_series.append(series_name)
            std_resid = fit_result.resid / fit_result.conditional_volatility
            standardized_residuals[series_name] = std_resid
            variance_forecasts[series_name] = var_forecast
        else:
            print(f"  -> GARCH for {series_name} failed. Excluding from covariance matrix.")

    if len(successful_series) < 2:
        raise RuntimeError("Covariance matrix requires at least 2 series with successful GARCH fits.")
        
    residuals_df = pd.DataFrame(standardized_residuals).reindex(model_input_returns.index).loc[:, successful_series]

    # --- 4. Multivariate DCC Fitting ---
    print(f"Fitting DCC(1,1) on {residuals_df.shape[1]} standardized residual series...")
    dcc_fit, corr_forecast = fit_dcc_model(residuals_df)
    
    if corr_forecast is None:
        raise RuntimeError("DCC model failed to converge. Cannot build covariance matrix.")

    corr_forecast_df = pd.DataFrame(corr_forecast, index=successful_series, columns=successful_series)

    # --- 5. Reconstruct Dynamic Covariance (H_t) ---
    vol_forecasts = np.sqrt(pd.Series(variance_forecasts))
    D_t = np.diag(vol_forecasts)
    H_t = D_t @ corr_forecast_df.values @ D_t
    H_t_df = pd.DataFrame(H_t, index=successful_series, columns=successful_series)

    # --- 6. Ledoit-Wolf Shrinkage Integration ---
    print("Applying Ledoit-Wolf shrinkage to stabilize the GARCH-DCC matrix...")
    # We shrink the GARCH-DCC forecast (H_t) towards a stable target,
    # using the historical returns to estimate the optimal shrinkage intensity (delta).
    shrunk_cov_matrix, shrinkage_delta = apply_ledoit_wolf_shrinkage(
        sample_cov=H_t_df.values,
        returns_array=model_input_returns[successful_series].values
    )
    final_shrunk_cov_full_df = pd.DataFrame(
        shrunk_cov_matrix, 
        index=successful_series, 
        columns=successful_series
    )
    
    # --- 7. Finalize and Drop FX Factor ---
    # As per spec, drop the FX factor row/col from the final matrix
    # that goes to the optimizer.
    final_asset_cov_df = final_shrunk_cov_full_df
    if fx_col_name and fx_col_name in final_asset_cov_df.columns:
        final_asset_cov_df = final_asset_cov_df.drop(index=fx_col_name, columns=fx_col_name)
        print(f"  -> Dropping '{fx_col_name}' from final optimizer covariance matrix.")
    
    print("Successfully built final covariance matrix.")
    
    diagnostics = {
        'num_series_in': model_input_returns.shape[1],
        'num_assets_out': final_asset_cov_df.shape[1],
        'shrinkage_intensity_delta': shrinkage_delta,
        'full_covariance_matrix_with_fx': final_shrunk_cov_full_df # Store for attribution
    }
    
    return final_asset_cov_df, diagnostics

if __name__ == '__main__':
    from ..config import load_config
    from pathlib import Path
    import shutil

    # Create a dummy config
    # ... (assuming previous test setup code for config) ...
    # ... as the RiskEngineConfig itself doesn't need to change ...
    
    # --- GIVEN ---
    np.random.seed(42)
    n_obs, n_assets = 500, 3
    dates = pd.date_range("2020-01-01", periods=n_obs)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    test_returns = pd.DataFrame(np.random.randn(n_obs, n_assets) * 0.02, index=dates, columns=tickers)
    
    # Create an FX series that is correlated with one of the assets
    fx_returns = pd.DataFrame(
        0.5 * test_returns['ASSET_1'] + np.random.randn(n_obs) * 0.005,
        columns=['USD_BRL_ret']
    )
    
    # Mock config object
    mock_config = RiskEngineConfig(
        garch={'dist': 'gaussian', 'min_obs': 250, 'refit_freq_days': 21},
        dcc={'a_init': 0.02, 'b_init': 0.97},
        shrinkage={'method': 'ledoit_wolf', 'floor': 0}
    )

    try:
        # --- WHEN ---
        # Run the engine with the FX factor
        sigma, diags = build_covariance_matrix(test_returns, mock_config, fx_returns_df=fx_returns)

        # --- THEN ---
        print("\n--- Output ---")
        print("Final Asset Covariance Matrix (Sigma) shape:", sigma.shape)
        print("Columns:", sigma.columns.tolist())
        
        # --- Validation ---
        # 1. Final matrix should only contain assets, not the FX factor
        assert 'USD_BRL_ret' not in sigma.columns
        assert sigma.shape == (n_assets, n_assets)
        
        # 2. The full matrix (with FX) should be available in diagnostics
        full_sigma = diags['full_covariance_matrix_with_fx']
        assert 'USD_BRL_ret' in full_sigma.columns
        assert full_sigma.shape == (n_assets + 1, n_assets + 1)
        
        # 3. Check the correlation
        # The correlation between ASSET_1 and USD_BRL should be high
        full_corr = np.linalg.inv(np.diag(np.sqrt(np.diag(full_sigma)))) @ full_sigma @ np.linalg.inv(np.diag(np.sqrt(np.diag(full_sigma))))
        fx_asset1_corr = full_corr.loc['ASSET_1', 'USD_BRL_ret']
        print(f"\nEstimated correlation between ASSET_1 and USD_BRL: {fx_asset1_corr:.4f}")
        assert fx_asset1_corr > 0.5
        
        print("\nOK: Risk engine correctly incorporates FX correlations and provides the correctly shaped final matrix.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()