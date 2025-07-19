import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Optional, List

def run_mean_variance_optimization(
    mu: pd.Series,
    sigma: pd.DataFrame,
    risk_aversion: float,
    constraints: List[cp.constraints.constraint.Constraint]
) -> Optional[pd.Series]:
    """
    Solves the mean-variance optimization problem for a given risk aversion.

    This function finds the portfolio weights 'w' that maximize the quadratic utility:
    Objective: w.T * mu - (gamma / 2) * w.T * Sigma * w
    where gamma is the risk aversion parameter.

    This is the standard formulation for mean-variance optimization and is a
    convex quadratic program, which can be solved efficiently.

    Args:
        mu: A pandas Series of expected excess returns.
        sigma: A pandas DataFrame of the asset covariance matrix.
        risk_aversion: The scalar risk aversion parameter (gamma). A higher
                       value leads to a more conservative (lower-risk) portfolio.
        constraints: A list of pre-constructed cvxpy constraint objects.

    Returns:
        A pandas Series of optimal portfolio weights, indexed by ticker.
        Returns None if the optimization fails.
    """
    # Ensure mu and sigma are aligned
    tickers = mu.index.tolist()
    sigma = sigma.reindex(index=tickers, columns=tickers)
    
    n = len(tickers)
    w = cp.Variable(n) # The portfolio weights vector to be solved for

    # Define the objective function
    expected_return = mu.values @ w
    risk_term = cp.quad_form(w, sigma.values) # Efficiently calculates w.T * Sigma * w
    
    utility = expected_return - (risk_aversion / 2) * risk_term
    objective = cp.Maximize(utility)
    
    # Define the problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    print("Solving mean-variance optimization problem...")
    try:
        # OSQP is a good, fast solver for QPs. ECOS is another option.
        # It's important to handle cases where the problem is infeasible or unbounded.
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Optimizer failed or found a non-optimal solution. Status: {problem.status}")
            return None
        
        optimal_weights = pd.Series(w.value, index=tickers, name="weights")
        # Due to numerical precision, clip very small negative weights to zero
        optimal_weights[optimal_weights < 0] = 0
        # Re-normalize to ensure sum-to-one constraint holds perfectly
        optimal_weights /= optimal_weights.sum()

        print("Successfully found optimal weights.")
        return optimal_weights

    except cp.SolverError as e:
        print(f"CVXPY SolverError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        return None

def build_basic_constraints(n_assets: int, w_variable: cp.Variable) -> list:
    """Helper function to create a standard set of long-only, fully-invested constraints."""
    return [
        cp.sum(w_variable) == 1, # Fully invested
        w_variable >= 0          # Long only (no shorting)
    ]

if __name__ == '__main__':
    print("--- Running Mean-Variance Optimizer Module Standalone Test ---")

    # --- GIVEN ---
    # 1. A 3-asset universe with clear characteristics
    tickers = ['HIGH_SHARPE', 'MID_SHARPE', 'LOW_SHARPE']
    
    # Asset 1 has high return, low vol (high Sharpe)
    # Asset 3 has low return, high vol (low Sharpe)
    mu_test = pd.Series([0.15, 0.10, 0.05], index=tickers)
    sigma_test = pd.DataFrame(
        [
            [0.02, 0.01, 0.005], # Vol of HIGH_SHARPE = sqrt(0.02) = 14%
            [0.01, 0.04, 0.015], # Vol of MID_SHARPE = sqrt(0.04) = 20%
            [0.005, 0.015, 0.06] # Vol of LOW_SHARPE = sqrt(0.06) = 24%
        ],
        columns=tickers, index=tickers
    )
    
    # 2. A moderate risk aversion. A value similar to the one calculated
    # from the market is a good starting point.
    gamma = 2.5
    
    # 3. Standard constraints: long-only, fully invested
    w_test = cp.Variable(len(tickers))
    constraints_test = build_basic_constraints(len(tickers), w_test)

    try:
        # --- WHEN ---
        optimal_w = run_mean_variance_optimization(mu_test, sigma_test, gamma, constraints_test)
        
        # --- THEN ---
        assert optimal_w is not None, "Optimization failed on a simple, valid problem."
        
        print("\n--- Optimizer Inputs ---")
        print("Expected Returns (mu):\n", mu_test)
        print("\nCovariance Matrix (Sigma):\n", sigma_test)
        print(f"\nRisk Aversion (gamma): {gamma}")
        
        print("\n--- Optimal Weights ---")
        print(optimal_w)
        
        # --- Validation ---
        # 1. Check if constraints are met
        assert np.isclose(optimal_w.sum(), 1.0), "Weights do not sum to 1."
        assert (optimal_w >= -1e-6).all(), "Negative weights found (violates long-only)." # Allow for tiny numerical error
        print("\nOK: Basic constraints (sum-to-one, long-only) are satisfied.")

        # 2. Check if the allocation makes economic sense
        # The asset with the highest Sharpe ratio should get the highest allocation.
        assert optimal_w.idxmax() == 'HIGH_SHARPE', "Allocation should be highest for the highest Sharpe asset."
        assert optimal_w['HIGH_SHARPE'] > optimal_w['MID_SHARPE'] > optimal_w['LOW_SHARPE']
        print("OK: Portfolio allocation is intuitive (favors higher Sharpe assets).")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()