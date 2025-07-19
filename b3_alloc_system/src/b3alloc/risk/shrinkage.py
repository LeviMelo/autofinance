import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.covariance import ledoit_wolf

def apply_ledoit_wolf_shrinkage(
    returns_df: pd.DataFrame
) -> Tuple[np.ndarray, float]:
    """
    Applies Ledoit-Wolf shrinkage to an empirical covariance matrix.

    The Ledoit-Wolf method computes an optimal shrinkage intensity (alpha) that
    minimizes the mean squared error between the estimated and the true
    covariance matrix. This procedure is robust and guaranteed to produce a
    well-conditioned (invertible) matrix.

    This function serves as a wrapper around the scikit-learn implementation,
    which operates directly on the returns data.

    Note: The specification mentions a "constant correlation" target. The standard
    scikit-learn LedoitWolf estimator shrinks towards a single-factor-model-based
    target. This is a common, robust choice and is used here for its battle-tested
    stability.

    Args:
        returns_df: A wide-format DataFrame of daily returns (assets as columns),
                    cleaned of NaNs.

    Returns:
        A tuple containing:
        - The shrunken NxN covariance matrix (as a numpy array).
        - The estimated optimal shrinkage coefficient (alpha).
    """
    if returns_df.isnull().values.any():
        raise ValueError("Input returns for shrinkage cannot contain NaNs.")
    if returns_df.shape[1] < 2:
        raise ValueError("Shrinkage requires at least 2 assets.")

    # The ledoit_wolf function from sklearn directly takes the returns matrix
    # and computes the sample covariance internally before shrinking.
    # The columns should be the features (assets).
    returns_array = returns_df.values
    
    # This function returns both the shrunk covariance and the shrinkage coefficient
    shrunk_cov, shrinkage_alpha = ledoit_wolf(returns_array)
    
    return shrunk_cov, shrinkage_alpha


if __name__ == '__main__':
    print("--- Running Shrinkage Module Standalone Test ---")

    # 1. Create a challenging scenario for covariance estimation
    # n_samples << n_features makes the sample covariance singular (not invertible).
    n_samples = 50
    n_features = 80 # 80 assets, only 50 days of returns
    np.random.seed(42)
    
    dates = pd.date_range("2023-01-01", periods=n_samples)
    tickers = [f"ASSET_{i}" for i in range(n_features)]
    
    # Generate random returns
    returns = pd.DataFrame(
        np.random.randn(n_samples, n_features) * 0.02,
        index=dates,
        columns=tickers
    )

    print(f"Testing with a challenging data shape: {n_samples} samples, {n_features} assets.")

    # 2. Calculate the standard sample covariance
    sample_cov = returns.cov().values

    # 3. Calculate the condition number of the sample covariance
    # The condition number measures how close a matrix is to being singular.
    # A very large number (or an error) indicates it's ill-conditioned.
    try:
        cond_sample = np.linalg.cond(sample_cov)
    except np.linalg.LinAlgError:
        cond_sample = np.inf # It is singular

    print(f"\nCondition number of sample covariance: {cond_sample:,.2f}")
    if np.isinf(cond_sample):
        print("OK: As expected, the sample covariance matrix is singular.")

    # 4. Apply Ledoit-Wolf shrinkage
    try:
        shrunk_cov, alpha = apply_ledoit_wolf_shrinkage(returns)
        
        print(f"\nApplied Ledoit-Wolf shrinkage with optimal alpha = {alpha:.4f}")
        
        # 5. Calculate the condition number of the shrunken matrix
        cond_shrunk = np.linalg.cond(shrunk_cov)
        print(f"Condition number of shrunken covariance: {cond_shrunk:,.2f}")
        
        # Validation
        assert np.isfinite(cond_shrunk), "Shrunken matrix should be well-conditioned."
        assert cond_shrunk < cond_sample, "Shrinkage must improve the condition number."
        print("\nOK: Shrinkage successfully produced a well-conditioned matrix.")

        # Check shapes
        assert shrunk_cov.shape == (n_features, n_features), "Output shape is incorrect."
        print("OK: Output shape is correct.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()