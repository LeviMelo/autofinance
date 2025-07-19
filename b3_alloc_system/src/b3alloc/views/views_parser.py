import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Tuple, Any

def parse_qualitative_views(
    view_definitions: List[Dict[str, Any]],
    universe: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a list of qualitative view dictionaries into Black-Litterman P and Q matrices.

    This function supports two types of views:
    1.  'absolute': "Asset X will return Y%".
    2.  'relative': "Asset X will outperform Asset Y by Z%".

    Args:
        view_definitions: A list of view dictionaries, typically loaded from a YAML file.
        universe: A sorted list of all ticker symbols in the investment universe.
                  The order of this list defines the column order of the P matrix.

    Returns:
        A tuple containing:
        - P (np.ndarray): The KxN pick matrix for the qualitative views.
        - Q (np.ndarray): The Kx1 vector of view values.
    """
    if sorted(universe) != universe:
        raise ValueError("The 'universe' list must be sorted alphabetically.")

    num_assets = len(universe)
    p_rows = []
    q_rows = []
    
    # Create a mapping from ticker to its index in the universe for quick lookups
    ticker_to_idx = {ticker: i for i, ticker in enumerate(universe)}

    print(f"Parsing {len(view_definitions)} qualitative views...")

    for i, view in enumerate(view_definitions):
        view_type = view.get('type')
        expression = view.get('expr')
        magnitude = view.get('magnitude')

        if not all([view_type, expression, isinstance(magnitude, (int, float))]):
            raise ValueError(f"View #{i+1} is malformed. It must have 'type', 'expr', and 'magnitude'.")
            
        p_row = np.zeros(num_assets)

        if view_type == 'absolute':
            # Absolute view: P has a 1 at the asset's position.
            ticker = expression.strip()
            if ticker not in ticker_to_idx:
                print(f"Warning: Ticker '{ticker}' in absolute view not in universe. Skipping view.")
                continue
            
            p_row[ticker_to_idx[ticker]] = 1.0
        
        elif view_type == 'relative':
            # Relative view: "A - B". P has a 1 at A's position and -1 at B's.
            try:
                ticker_a, ticker_b = [t.strip() for t in expression.split('–')] # Note: using en-dash
                if len(ticker_a) == 0 or len(ticker_b) == 0: # Check for malformed split
                    ticker_a, ticker_b = [t.strip() for t in expression.split('-')]
            except ValueError:
                raise ValueError(f"Relative view '{expression}' is malformed. Expected 'TICKER_A – TICKER_B'.")

            if ticker_a not in ticker_to_idx or ticker_b not in ticker_to_idx:
                print(f"Warning: One or both tickers in relative view '{expression}' not in universe. Skipping view.")
                continue
            
            p_row[ticker_to_idx[ticker_a]] = 1.0
            p_row[ticker_to_idx[ticker_b]] = -1.0
            
        else:
            raise ValueError(f"Unknown view type '{view_type}' in view #{i+1}.")

        p_rows.append(p_row)
        q_rows.append([magnitude])

    if not p_rows:
        # Return empty matrices if no valid views were parsed
        return np.empty((0, num_assets)), np.empty((0, 1))

    return np.array(p_rows), np.array(q_rows)


if __name__ == '__main__':
    print("--- Running Qualitative Views Parser Module Standalone Test ---")

    # --- GIVEN ---
    # 1. A sorted investment universe
    test_universe = sorted(['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'MGLU3.SA'])
    
    # 2. A YAML-like structure of qualitative views
    yaml_content = """
    views:
      - type: relative
        expr: "VALE3.SA – PETR4.SA"
        magnitude: 0.03    # Expect VALE3 to outperform PETR4 by 3%
        confidence: 0.70
      - type: absolute
        expr: "MGLU3.SA"
        magnitude: -0.05   # Expect MGLU3 to have a return of -5%
        confidence: 0.50
      - type: relative
        expr: "ITUB4.SA - MGLU3.SA" # Using hyphen instead of en-dash
        magnitude: 0.06
        confidence: 0.80
      - type: absolute
        expr: "NON_EXISTENT.SA" # A ticker not in our universe
        magnitude: 0.10
        confidence: 0.40
    """
    view_defs = yaml.safe_load(yaml_content)['views']
    
    # --- WHEN ---
    P_qual, Q_qual = parse_qualitative_views(view_defs, test_universe)
    
    # --- THEN ---
    print("\nParsed P matrix (Qualitative):\n", P_qual)
    print("\nParsed Q vector (Qualitative):\n", Q_qual)
    
    # --- Validation ---
    # Universe order: ITUB4, MGLU3, PETR4, VALE3
    # Expected P matrix (3 valid views x 4 assets)
    expected_P = np.array([
    # View 1: VALE3 - PETR4
       [0.,  0., -1., 1.],
    # View 2: MGLU3
       [0.,  1.,  0., 0.],
    # View 3: ITUB4 - MGLU3
       [1., -1.,  0., 0.]
    ])
    
    # Expected Q vector
    expected_Q = np.array([[0.03], [-0.05], [0.06]])
    
    assert P_qual.shape == (3, 4), f"P matrix shape is {P_qual.shape}, expected (3, 4)."
    assert Q_qual.shape == (3, 1), "Q vector shape is incorrect."
    
    assert np.allclose(P_qual, expected_P), "P matrix content is incorrect."
    assert np.allclose(Q_qual, expected_Q), "Q vector content is incorrect."
    
    print("\nOK: Parser correctly translates qualitative views into P and Q matrices.")