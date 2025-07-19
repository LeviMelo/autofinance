import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.covariance import ledoit_wolf

def apply_ledoit_wolf_shrinkage(
    sample_cov: np.ndarray,
    returns_array: np.ndarray,
    shrinkage_target: str = 'constant_correlation'
) -> Tuple[np.ndarray, float]:
    """
    Applies Ledoit-Wolf shrinkage to an empirical covariance matrix.

    This implementation follows the direct formula for shrinkage, allowing the
    GARCH-DCC covariance forecast to be used as the "sample_cov".

    Args:
        sample_cov: The NxN sample covariance matrix (e.g., from GARCH-DCC).
        returns_array: The NxT array of returns used to estimate the shrinkage intensity.
        shrinkage_target: The structured matrix to shrink towards.
                         'constant_correlation' is specified in the blueprint.

    Returns:
        A tuple containing:
        - The shrunken NxN covariance matrix (as a numpy array).
        - The estimated optimal shrinkage coefficient (delta).
    """
    if sample_cov.shape[0] != sample_cov.shape[1]:
        raise ValueError("Sample covariance must be a square matrix.")
    if sample_cov.shape[0] != returns_array.shape[1]:
        raise ValueError("Dimensions of sample_cov and returns_array do not match.")

    n_assets = sample_cov.shape[0]
    n_obs = returns_array.shape[0]

    # --- 1. Define the shrinkage target (F) ---
    if shrinkage_target == 'constant_correlation':
        asset_variances = np.diag(sample_cov).reshape(-1, 1)
        sqrt_var = np.sqrt(asset_variances)
        corr_matrix = sample_cov / (sqrt_var @ sqrt_var.T)
        
        # Average correlation
        upper_triangle_indices = np.triu_indices(n_assets, k=1)
        avg_corr = np.mean(corr_matrix[upper_triangle_indices])
        
        # Build target matrix F
        F = np.full((n_assets, n_assets), avg_corr)
        np.fill_diagonal(F, 1.0)
        F = np.diag(np.sqrt(asset_variances).flatten()) @ F @ np.diag(np.sqrt(asset_variances).flatten())
    else:
        raise NotImplementedError(f"Shrinkage target '{shrinkage_target}' not implemented.")

    # --- 2. Estimate the shrinkage intensity (delta) ---
    # This is a simplified version of the formula from Ledoit & Wolf (2004)
    # delta = sum(Var(s_ij)) / sum((s_ij - f_ij)^2)
    
    # Estimate pi-hat (sum of variances of sample covariance entries)
    y_t = returns_array
    y_t_centered = y_t - y_t.mean(axis=0)
    pi_hat_mat = np.zeros_like(sample_cov)
    for t in range(n_obs):
        pi_hat_mat += np.outer(y_t_centered[t]**2, y_t_centered[t]**2)
    pi_hat = np.sum(pi_hat_mat / n_obs - sample_cov**2)

    # Estimate rho-hat (sum of squared errors between sample and target)
    rho_hat = np.sum((sample_cov - F)**2)
    
    # Shrinkage constant
    delta = pi_hat / rho_hat
    delta = np.clip(delta, 0, 1) # Ensure delta is between 0 and 1

    # --- 3. Apply shrinkage ---
    shrunk_cov = (1 - delta) * sample_cov + delta * F
    
    return shrunk_cov, delta


if __name__ == '__main__':
    print("--- Running Shrinkage Module Standalone Test ---")

    # 1. Create a challenging scenario for covariance estimation
    n_samples = 50
    n_features = 80
    np.random.seed(42)
    
    returns_array = np.random.randn(n_samples, n_features) * 0.02
    
    # Calculate the empirical covariance matrix, which will be ill-conditioned
    sample_cov_ill = np.cov(returns_array, rowvar=False)

    print(f"Testing with a challenging data shape: {n_samples} samples, {n_features} assets.")
    
    try:
        cond_sample = np.linalg.cond(sample_cov_ill)
    except np.linalg.LinAlgError:
        cond_sample = np.inf
    print(f"\nCondition number of original matrix: {cond_sample:,.2f}")
    assert np.isinf(cond_sample), "Test setup should result in a singular matrix."

    # 4. Apply the direct Ledoit-Wolf shrinkage
    try:
        shrunk_cov, delta = apply_ledoit_wolf_shrinkage(sample_cov_ill, returns_array)
        
        print(f"\nApplied Ledoit-Wolf shrinkage with optimal delta = {delta:.4f}")
        
        # 5. Calculate the condition number of the shrunken matrix
        cond_shrunk = np.linalg.cond(shrunk_cov)
        print(f"Condition number of shrunken covariance: {cond_shrunk:,.2f}")
        
        assert np.isfinite(cond_shrunk), "Shrunken matrix should be well-conditioned."
        assert cond_shrunk < 1e6, "Condition number should be significantly improved." # Check it's not still huge
        print("\nOK: Shrinkage successfully produced a well-conditioned matrix.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()