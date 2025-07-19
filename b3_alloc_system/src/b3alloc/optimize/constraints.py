import cvxpy as cp
import pandas as pd
from typing import List, Dict, Optional
from ..config import OptimizerConfig

def build_optimizer_constraints(
    w_variable: cp.Variable,
    config: OptimizerConfig,
    asset_universe: List[str],
    sector_map: Optional[Dict[str, str]] = None
) -> List[cp.constraints.constraint.Constraint]:
    """
    Builds a list of cvxpy constraints based on the provided configuration.

    Args:
        w_variable: The cvxpy Variable representing the portfolio weights.
        config: The OptimizerConfig Pydantic model.
        asset_universe: The list of assets in the order they appear in w_variable.
        sector_map: A dictionary mapping tickers to their sectors, required for
                    sector constraints.

    Returns:
        A list of cvxpy constraint objects.
    """
    constraints = []
    
    # --- Core Constraints ---
    constraints.append(cp.sum(w_variable) == 1)
    
    if config.long_only:
        constraints.append(w_variable >= 0)
        
    # --- Weight Cap Constraints ---
    if config.name_cap is not None:
        constraints.append(w_variable <= config.name_cap)
    
    if config.sector_cap is not None:
        if not sector_map:
            raise ValueError("A sector_map is required for sector cap constraints.")
            
        sectors = {}
        for i, ticker in enumerate(asset_universe):
            sector = sector_map.get(ticker, 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(w_variable[i])
            
        for sector, weights_in_sector in sectors.items():
            if sector != 'Unknown':
                constraints.append(cp.sum(weights_in_sector) <= config.sector_cap)

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
    # Use the Pydantic model for config
    test_config = OptimizerConfig(
        objective='max_sharpe', # Objective is not used here, but required by model
        long_only=True,
        name_cap=0.10,
        sector_cap=0.15,
        turnover_penalty_bps=0
    )
    w = cp.Variable(len(test_universe))
    
    # --- WHEN ---
    try:
        constraints_list = build_optimizer_constraints(w, test_config, test_universe, test_sector_map)
        
        # --- THEN ---
        print(f"\nGenerated {len(constraints_list)} constraint objects.")
        
        # Validation
        assert len(constraints_list) == 6, "Incorrect number of constraints generated."
        
        print("\nOK: Correct number of constraints created.")
        
        sector_constraint_str = str(constraints_list[-1])
        assert '<=' in sector_constraint_str
        assert str(test_config.sector_cap) in sector_constraint_str
        print("OK: Sector constraint appears to be correctly formulated.")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()