import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from ..views.views_parser import parse_qualitative_views

def build_absolute_views(
    model_views: Dict[str, pd.Series],
    universe: List[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Constructs the Black-Litterman P and Q matrices for absolute, model-driven views.
    (This function remains largely the same but is now a component of the full builder).
    """
    # Ensure the universe is sorted for consistent matrix construction
    universe = sorted(universe)

    p_matrices = []
    q_vectors = []
    num_assets = len(universe)
    
    for view_name, view_series in model_views.items():
        if view_series is None or view_series.empty:
            continue

        aligned_view = view_series.reindex(universe)
        if aligned_view.isnull().any():
            missing = aligned_view[aligned_view.isnull()].index.tolist()
            raise ValueError(f"Model view '{view_name}' is missing forecasts for: {missing}")

        p_matrices.append(np.eye(num_assets))
        q_vectors.append(aligned_view.values.reshape(-1, 1))

    if not p_matrices:
        return None, None

    return np.vstack(p_matrices), np.vstack(q_vectors)


def build_full_view_matrices(
    model_views: Dict[str, pd.Series],
    qualitative_view_defs: Optional[List[Dict[str, Any]]],
    universe: List[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Orchestrates the construction of the complete P and Q matrices from all
    view sources (model-based and qualitative).

    Args:
        model_views: Dictionary of programmatic views (e.g., from FF and VAR models).
        qualitative_view_defs: A list of qualitative view dictionaries from a config file.
        universe: A sorted list of all ticker symbols in the investment universe.

    Returns:
        A tuple containing the final, stacked P and Q matrices.
    """
    print("Building full P and Q matrices from all view sources...")
    all_p = []
    all_q = []

    # 1. Process model-driven absolute views
    P_model, Q_model = build_absolute_views(model_views, universe)
    if P_model is not None and Q_model is not None:
        print(f"  -> Added {P_model.shape[0]} model-driven views.")
        all_p.append(P_model)
        all_q.append(Q_model)

    # 2. Process user-defined qualitative views
    if qualitative_view_defs:
        P_qual, Q_qual = parse_qualitative_views(qualitative_view_defs, universe)
        if P_qual.shape[0] > 0:
            print(f"  -> Added {P_qual.shape[0]} qualitative views.")
            all_p.append(P_qual)
            all_q.append(Q_qual)

    if not all_p:
        print("Warning: No valid views were processed. Returning empty matrices.")
        return None, None

    # 3. Stack all matrices together
    P_final = np.vstack(all_p)
    Q_final = np.vstack(all_q)
    
    print(f"Successfully constructed final P matrix with shape {P_final.shape} and Q vector with shape {Q_final.shape}.")
    return P_final, Q_final


if __name__ == '__main__':
    print("--- Running Full View Builder Module Standalone Test ---")

    # --- GIVEN ---
    test_universe = sorted(['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'])
    
    # 1. Model views
    model_views_test = {
        'ff_view': pd.Series({'PETR4.SA': 0.10, 'VALE3.SA': 0.12, 'ITUB4.SA': 0.08})
    }
    
    # 2. Qualitative views
    qual_views_test = [
        {'type': 'relative', 'expr': 'VALE3.SA – PETR4.SA', 'magnitude': 0.03, 'confidence': 0.7},
        {'type': 'absolute', 'expr': 'ITUB4.SA', 'magnitude': 0.09, 'confidence': 0.9}
    ]
    
    # --- WHEN ---
    P, Q = build_full_view_matrices(model_views_test, qual_views_test, test_universe)
    
    # --- THEN ---
    print("\nFinal P Matrix (Stacked):\n", P)
    print("\nFinal Q Vector (Stacked):\n", Q)
    
    # --- Validation ---
    # Universe order should be sorted: ITUB4.SA, PETR4.SA, VALE3.SA
    
    # Expected P from models (3 views)
    P_exp_model = np.eye(3)
    
    # Expected P from qualitative views (2 views)
    P_exp_qual = np.array([
        [0., -1., 1.], # VALE3 - PETR4
        [1.,  0., 0.]  # ITUB4
    ])
    
    expected_P = np.vstack([P_exp_model, P_exp_qual])
    
    # Expected Q from models
    # Reordered: ITUB4=0.08, PETR4=0.10, VALE3=0.12
    Q_exp_model = np.array([[0.08], [0.10], [0.12]])
    
    # Expected Q from qualitative views
    Q_exp_qual = np.array([[0.03], [0.09]])
    
    expected_Q = np.vstack([Q_exp_model, Q_exp_qual])

    assert P.shape == (5, 3), "Final P matrix has incorrect shape."
    assert Q.shape == (5, 1), "Final Q vector has incorrect shape."
    
    assert np.allclose(P, expected_P), "Final P matrix content is incorrect."
    assert np.allclose(Q, expected_Q), "Final Q vector content is incorrect."
    
    print("\nOK: Model-driven and qualitative views were correctly stacked into final P and Q.")

    # --- Test with unsorted universe ---
    print("\n--- Testing with unsorted universe ---")
    unsorted_universe = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    P_unsorted, Q_unsorted = build_full_view_matrices(model_views_test, qual_views_test, unsorted_universe)
    assert np.allclose(P_unsorted, expected_P), "P matrix is incorrect when universe is unsorted."
    assert np.allclose(Q_unsorted, expected_Q), "Q vector is incorrect when universe is unsorted."
    print("OK: View builder correctly handles unsorted universe input.")