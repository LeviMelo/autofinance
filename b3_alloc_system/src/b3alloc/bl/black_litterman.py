import pandas as pd
import numpy as np
from typing import Tuple, Dict

from ..config import BlackLittermanConfig

def calculate_risk_aversion(market_excess_returns: pd.Series) -> float:
    """
    Calculates the market-implied risk aversion parameter (lambda) from historical
    market data. Lambda = Market Sharpe Ratio / Market Volatility.

    This simplifies to: E[R_m - R_f] / Var(R_m - R_f)

    Args:
        market_excess_returns: A Series of historical daily market excess returns.

    Returns:
        The scalar risk aversion parameter, lambda.
    """
    if market_excess_returns.empty:
        raise ValueError("Market excess returns cannot be empty.")
        
    mean_excess_return = market_excess_returns.mean()
    var_excess_return = market_excess_returns.var()
    
    if var_excess_return < 1e-12:
        # Handle case of zero variance to avoid division by zero
        return 2.0 # Return a default, sensible value
        
    risk_aversion = mean_excess_return / var_excess_return
    return risk_aversion

def calculate_equilibrium_returns(
    risk_aversion: float,
    sigma_matrix: pd.DataFrame,
    market_cap_weights: pd.Series
) -> pd.Series:
    """
    Calculates the market equilibrium prior returns (Pi) based on reverse optimization.

    Formula: Pi = lambda * Sigma * w_mkt

    Args:
        risk_aversion: The scalar risk aversion parameter (lambda).
        sigma_matrix: The NxN covariance matrix of asset returns.
        market_cap_weights: A Series of market-cap weights for the assets.

    Returns:
        A Series of annualized equilibrium expected excess returns.
    """
    # Align weights and sigma matrix
    aligned_weights = market_cap_weights.reindex(sigma_matrix.columns).fillna(0)
    
    pi_vector = risk_aversion * sigma_matrix.dot(aligned_weights)
    pi_series = pd.Series(pi_vector, index=sigma_matrix.columns, name="pi_prior")
    
    return pi_series

def calculate_posterior_returns(
    mu_prior: pd.Series,
    sigma_matrix: pd.DataFrame,
    p_matrix: np.ndarray,
    q_vector: np.ndarray,
    omega_matrix: np.ndarray,
    tau: float
) -> pd.Series:
    """
    Calculates the Black-Litterman posterior expected returns (mu_BL).

    This function implements the core BL formula using the more stable matrix form:
    mu_post = inv(inv(tau*Sigma) + P'*inv(Omega)*P) * (inv(tau*Sigma)*Pi + P'*inv(Omega)*Q)

    Args:
        mu_prior: The vector of prior equilibrium returns (Pi).
        sigma_matrix: The covariance matrix of asset returns (Sigma).
        p_matrix: The pick/link matrix for views (P).
        q_vector: The vector of view returns (Q).
        omega_matrix: The diagonal covariance matrix of view errors (Omega).
        tau: A scalar controlling the uncertainty of the prior (tau).

    Returns:
        A Series containing the final posterior expected excess returns.
    """
    print("Calculating Black-Litterman posterior returns...")
    # Ensure consistent ordering
    asset_names = mu_prior.index
    sigma_matrix = sigma_matrix.reindex(index=asset_names, columns=asset_names)
    
    # 1. Pre-compute inverses for stability and clarity
    # Add a small jitter to Omega's diagonal to ensure invertibility, just in case
    omega_inv = np.linalg.inv(omega_matrix + np.eye(omega_matrix.shape[0]) * 1e-12)
    
    tau_sigma = tau * sigma_matrix.values
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    
    # 2. Calculate the two main terms of the BL formula
    # First term: The precision (inverse covariance) of the posterior distribution
    posterior_precision = tau_sigma_inv + p_matrix.T @ omega_inv @ p_matrix

    # Second term: The weighted average of the prior and the views
    weighted_priors_and_views = (tau_sigma_inv @ mu_prior.values.reshape(-1, 1)) + (p_matrix.T @ omega_inv @ q_vector)
    
    # 3. Calculate the posterior mean
    posterior_mean_vector = np.linalg.inv(posterior_precision) @ weighted_priors_and_views
    
    # 4. Format the output
    mu_posterior = pd.Series(
        posterior_mean_vector.flatten(), 
        index=asset_names,
        name="mu_posterior"
    )
    
    print("Successfully calculated posterior returns.")
    return mu_posterior

if __name__ == '__main__':
    print("--- Running Black-Litterman Module Standalone Test ---")

    # --- GIVEN ---
    # 1. A 2-asset universe
    tickers = ['ASSET_A', 'ASSET_B']
    sigma = pd.DataFrame(
        [[0.02, 0.01], [0.01, 0.03]], 
        columns=tickers, index=tickers
    )
    market_caps = pd.Series([0.6, 0.4], index=tickers)
    market_returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005) # Dummy history
    
    # 2. Hyperparameters
    tau = 0.05
    
    # 3. A single, absolute view on Asset A
    # "Asset A will have an annualized excess return of 10%"
    P = np.array([[1.0, 0.0]]) # Picks Asset A
    Q = np.array([[0.10]])      # The view value
    
    # We are very confident in this view, so Omega has a small variance
    Omega = np.array([[0.001]])

    try:
        # --- WHEN ---
        # A. Calculate intermediate inputs
        lmbda = calculate_risk_aversion(market_returns)
        pi = calculate_equilibrium_returns(lmbda, sigma, market_caps)
        
        # B. Calculate the final posterior
        mu_bl = calculate_posterior_returns(pi, sigma, P, Q, Omega, tau)
        
        # --- THEN ---
        print("\n--- Inputs to BL ---")
        print(f"Calculated Lambda: {lmbda:.4f}")
        print("Prior Returns (Pi):\n", pi)
        
        print("\n--- Output from BL ---")
        print("Posterior Returns (mu_BL):\n", mu_bl)
        
        # --- Validation ---
        # The posterior return for Asset A should have moved from its prior (pi)
        # towards the view (Q=0.10).
        pi_A = pi['ASSET_A']
        mu_bl_A = mu_bl['ASSET_A']
        view_Q = Q[0,0]
        
        print(f"\nPrior for Asset A:      {pi_A:.4%}")
        print(f"View for Asset A:       {view_Q:.4%}")
        print(f"Posterior for Asset A:  {mu_bl_A:.4%}")
        
        # Check if the posterior is between the prior and the view
        assert (mu_bl_A > pi_A and mu_bl_A < view_Q) or \
               (mu_bl_A < pi_A and mu_bl_A > view_Q)
        print("\nOK: Posterior for Asset A has shifted from the prior towards the view.")

        # The posterior for Asset B should also have shifted due to its covariance with A.
        pi_B = pi['ASSET_B']
        mu_bl_B = mu_bl['ASSET_B']
        
        assert not np.isclose(pi_B, mu_bl_B)
        print("OK: Posterior for Asset B has also been updated.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()