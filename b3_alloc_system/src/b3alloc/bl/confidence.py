import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from ..config import BlackLittermanConfig

def _get_ff_view_variances(
    ff_view: pd.Series,
    ff_diagnostics: Dict,
    config: BlackLittermanConfig
) -> pd.Series:
    """Calculates variances for the Fama-French model views."""
    if config.confidence.method == "rmse_based":
        if 'residual_variance' not in ff_diagnostics:
            raise ValueError("RMSE-based confidence requires 'residual_variance' in ff_view diagnostics.")
        res_var = ff_diagnostics['residual_variance']
        variances = res_var.reindex(ff_view.index).fillna(res_var.mean())
    else: # user_scaled
        variances = pd.Series(np.var(ff_view), index=ff_view.index)
        
    return variances * config.confidence.factor_scaler

def _get_var_view_variances(
    var_view: pd.Series,
    var_diagnostics: Dict,
    config: BlackLittermanConfig
) -> pd.Series:
    """Calculates variances for the VAR model views."""
    if config.confidence.method == "rmse_based":
        if 'sigma_u' not in var_diagnostics:
             raise ValueError("RMSE-based confidence requires 'sigma_u' in var_view diagnostics.")
        sigma_u = var_diagnostics['sigma_u']
        variances = pd.Series(np.diag(sigma_u), index=var_view.index)
    else: # user_scaled
        variances = pd.Series(np.var(var_view), index=var_view.index)
        
    return variances * config.confidence.var_scaler

def _get_qualitative_view_variances(
    qualitative_views: List[Dict],
    p_matrix_qual: np.ndarray,
    sigma: pd.DataFrame
) -> np.ndarray:
    """
    Calculates variances for qualitative views based on specified confidence.
    Follows the common Idzorek (2005) method where uncertainty is proportional
    to the variance of the portfolio defined by the view.
    """
    variances = []
    for i, view in enumerate(qualitative_views):
        p_row = p_matrix_qual[i, :]
        variance_of_view_portfolio = p_row.T @ sigma.values @ p_row
        
        # Heuristic: Confidence of 1.0 = variance of view.
        # Confidence of 0.0 = infinite variance.
        confidence = view.get('confidence', 0.5) # Default to 50% confidence
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Qualitative view confidence must be between 0 and 1.")
            
        # The less confident, the higher the variance
        # This is one of many possible heuristics.
        uncertainty = variance_of_view_portfolio / confidence
        variances.append(uncertainty)
        
    return np.array(variances)


def estimate_view_uncertainty(
    model_views: Dict[str, pd.Series],
    qualitative_views: Optional[List[Dict]],
    view_diagnostics: Dict[str, Dict],
    config: BlackLittermanConfig,
    p_matrix_qual: Optional[np.ndarray],
    sigma_matrix: pd.DataFrame
) -> np.ndarray:
    """
    Estimates the complete view uncertainty matrix (Omega) for all view types.
    """
    print(f"Estimating view uncertainty (Omega) using '{config.confidence.method}' method...")
    
    all_variances = []
    
    # --- Model-driven view uncertainty ---
    if 'ff_view' in model_views and model_views['ff_view'] is not None:
        ff_vars = _get_ff_view_variances(model_views['ff_view'], view_diagnostics.get('ff_view', {}), config)
        all_variances.append(ff_vars)

    if 'var_view' in model_views and model_views['var_view'] is not None:
        var_vars = _get_var_view_variances(model_views['var_view'], view_diagnostics.get('var_view', {}), config)
        all_variances.append(var_vars)

    # --- Qualitative view uncertainty ---
    if qualitative_views and p_matrix_qual is not None and p_matrix_qual.shape[0] > 0:
        qual_vars = _get_qualitative_view_variances(qualitative_views, p_matrix_qual, sigma_matrix)
        all_variances.append(pd.Series(qual_vars))

    if not all_variances:
        raise ValueError("No views provided to estimate uncertainty.")
        
    final_variances = pd.concat(all_variances)
    omega_matrix = np.diag(final_variances.values)

    print(f"Successfully constructed Omega matrix with shape {omega_matrix.shape}.")
    return omega_matrix


if __name__ == '__main__':
    # Use a mock/in-memory config instead of file I/O to simplify the test.
    from pydantic import BaseModel

    class MockConfidenceConfig(BaseModel):
        method: str = "rmse_based"
        factor_scaler: float = 1.0
        var_scaler: float = 2.0

    class MockBLConfig(BaseModel):
        tau: float = 0.05
        confidence: MockConfidenceConfig = MockConfidenceConfig()

    config = MockBLConfig()

    # --- Test Data ---
    tickers = ['ASSET_A', 'ASSET_B', 'ASSET_C']
    
    # Dummy views
    ff_view = pd.Series([0.08, 0.06, 0.05], index=tickers)
    var_view = pd.Series([0.07, 0.09, 0.08], index=tickers)
    model_views = {'ff_view': ff_view, 'var_view': var_view}
    
    # Dummy diagnostics for model views
    diagnostics = {
        'ff_view': {'residual_variance': pd.Series([0.005, 0.004, 0.0045], index=tickers)},
        'var_view': {'sigma_u': np.array([[0.003, 0.001, 0.001], 
                                          [0.001, 0.002, 0.001],
                                          [0.001, 0.001, 0.0025]])}
    }

    # Dummy qualitative views and associated matrices
    qual_views = [
        {'view': {'type': 'absolute', 'ticker': 'ASSET_A', 'expected_return': 0.10, 'confidence': 0.7}},
        {'view': {'type': 'relative', 'ticker1': 'ASSET_B', 'ticker2': 'ASSET_C', 'expected_difference': 0.02, 'confidence': 0.5}}
    ]
    # P matrix would be generated by ViewBuilder, we mock it here
    p_matrix_qual = np.array([
        [1, 0, 0],
        [0, 1, -1]
    ])
    sigma_matrix = pd.DataFrame(diagnostics['var_view']['sigma_u'], index=tickers, columns=tickers)
    
    print("--- Running Confidence Module Standalone Test ---")
    
    try:
        omega = estimate_view_uncertainty(
            model_views, 
            qual_views, 
            diagnostics, 
            config, 
            p_matrix_qual, 
            sigma_matrix
        )
        
        print("\n--- Output Omega Matrix (Combined) ---")
        print(omega)
        
        # Validation
        num_model_views = sum(v.size for v in model_views.values())
        num_qual_views = len(qual_views)
        expected_size = num_model_views + num_qual_views
        
        assert omega.shape == (expected_size, expected_size), f"Omega matrix shape should be ({expected_size}, {expected_size})"
        assert np.count_nonzero(omega - np.diag(np.diagonal(omega))) == 0, "Omega must be diagonal."
        
        # Check that variances are positive
        assert np.all(np.diagonal(omega) > 0), "All diagonal elements of Omega must be positive."

        print(f"\nOK: Combined Omega matrix ({expected_size}x{expected_size}) looks correct.")

        # Test with only model views
        omega_models_only = estimate_view_uncertainty(model_views, None, diagnostics, config, None, sigma_matrix)
        assert omega_models_only.shape == (num_model_views, num_model_views), "Model-only Omega shape is incorrect."
        print("\nOK: Model-only Omega matrix is correctly constructed.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()