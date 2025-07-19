import cvxpy as cp
import pandas as pd
from typing import List, Dict

def build_optimizer_constraints(
    w_variable: cp.Variable,
    config: Dict, # Expects the 'optimizer' section of the config
    asset_universe: List[str],
    sector_map: Dict[str, str] = None
) -> List[cp.constraints.constraint.Constraint]:
    """
    Builds a list of cvxpy constraints based on the provided configuration.

    Args:
        w_variable: The cvxpy Variable representing the portfolio weights.
        config: The optimizer configuration dictionary.
        asset_universe: The list of assets in the order they appear in w_variable.
        sector_map: A dictionary mapping tickers to their sectors, required for
                    sector constraints.

    Returns:
        A list of cvxpy constraint objects.
    """
    constraints = []
    n_assets = len(asset_universe)

    # --- Core Constraints ---
    # Budget constraint: weights must sum to 1
    constraints.append(cp.sum(w_variable) == 1)
    
    # Long-only constraint
    if config.get('long_only', True):
        constraints.append(w_variable >= 0)
        
    # --- Weight Cap Constraints ---
    # Single name cap
    if 'name_cap' in config:
        constraints.append(w_variable <= config['name_cap'])
    
    # Sector cap
    if 'sector_cap' in config:
        if not sector_map:
            raise ValueError("A sector_map is required for sector cap constraints.")
            
        # Group assets by sector
        sectors = {}
        for i, ticker in enumerate(asset_universe):
            sector = sector_map.get(ticker, 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(w_variable[i])
            
        # Add a sum constraint for each sector
        for sector, weights_in_sector in sectors.items():
            if sector != 'Unknown':
                constraints.append(cp.sum(weights_in_sector) <= config['sector_cap'])

    print(f"Built {len(constraints)} constraints for the optimizer.")
    return constraints


if __name__ == '__main__':
    print("--- Running Constraints Module Standalone Test ---")
    
    # --- GIVEN ---
    test_universe = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA']
    test_sector_map = {
        'PETR4.SA': 'Energy',
        'VALE3.SA': 'Materials',
        'ITUB4.SA': 'Financials',
        'BBDC4.SA': 'Financials'
    }
    test_config = {
        'long_only': True,
        'name_cap': 0.10,
        'sector_cap': 0.15
    }
    w = cp.Variable(len(test_universe))
    
    # --- WHEN ---
    try:
        constraints_list = build_optimizer_constraints(w, test_config, test_universe, test_sector_map)
        
        # --- THEN ---
        print(f"\nGenerated {len(constraints_list)} constraint objects.")
        
        # Validation
        # Expected constraints: sum=1, w>=0, w<=0.10, sum(financials)<=0.15, sum(materials)<=0.15, sum(energy)<=0.15
        # Total = 1 + 1 + 1 + 3 = 6
        assert len(constraints_list) == 6, "Incorrect number of constraints generated."
        
        print("\nOK: Correct number of constraints created.")
        
        # Check the structure of one constraint
        sector_constraint = str(constraints_list[-1])
        # Example output: 'sum(varX[0]) <= 0.15'
        assert '<=' in sector_constraint
        assert str(test_config['sector_cap']) in sector_constraint
        print("OK: Sector constraint appears to be correctly formulated.")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()