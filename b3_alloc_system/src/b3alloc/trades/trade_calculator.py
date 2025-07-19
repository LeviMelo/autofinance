import pandas as pd
import numpy as np
from typing import Literal

def calculate_target_shares(
    target_weights: pd.Series,
    portfolio_value: float,
    last_prices: pd.Series
) -> pd.DataFrame:
    """
    Converts ideal portfolio weights into target integer share counts.

    Args:
        target_weights: A Series of optimal asset weights from the optimizer.
        portfolio_value: The total current market value of the portfolio.
        last_prices: A Series of the most recent closing prices for each asset.

    Returns:
        A DataFrame with columns ['ticker', 'target_value', 'target_shares_frac',
        'last_price'] for each asset in the target portfolio.
    """
    if portfolio_value <= 0:
        return pd.DataFrame()

    # Align inputs to ensure consistent ordering and handling of assets
    weights, prices = target_weights.align(last_prices, join='inner')
    
    # Calculate the ideal monetary value for each asset
    target_value = weights * portfolio_value
    
    # Calculate the ideal fractional number of shares
    target_shares_frac = target_value / prices

    results_df = pd.DataFrame({
        'target_value': target_value,
        'target_shares_frac': target_shares_frac,
        'last_price': prices
    }).reset_index().rename(columns={'index': 'ticker'})
    
    return results_df


def resolve_fractional_shares(
    target_shares_df: pd.DataFrame,
    rounding_mode: Literal['round', 'floor', 'ceil'] = 'round'
) -> pd.DataFrame:
    """
    Rounds fractional target shares to the nearest integer.

    Args:
        target_shares_df: The DataFrame from `calculate_target_shares`.
        rounding_mode: The method for rounding ('round', 'floor', or 'ceil').

    Returns:
        The input DataFrame with an added 'target_shares' integer column.
    """
    if 'target_shares_frac' not in target_shares_df.columns:
        raise ValueError("Input DataFrame must contain 'target_shares_frac' column.")

    if rounding_mode == 'round':
        target_shares_df['target_shares'] = np.round(target_shares_df['target_shares_frac']).astype(int)
    elif rounding_mode == 'floor':
        target_shares_df['target_shares'] = np.floor(target_shares_df['target_shares_frac']).astype(int)
    elif rounding_mode == 'ceil':
        target_shares_df['target_shares'] = np.ceil(target_shares_df['target_shares_frac']).astype(int)
    else:
        raise ValueError(f"Unknown rounding mode: {rounding_mode}")
        
    return target_shares_df


def compute_trade_list(
    current_shares: pd.Series,
    target_shares: pd.Series
) -> pd.DataFrame:
    """
    Generates a list of trades required to move from current to target positions.

    Args:
        current_shares: A Series of current share holdings, indexed by ticker.
        target_shares: A Series of target share holdings, indexed by ticker.

    Returns:
        A DataFrame with columns ['ticker', 'current_shares', 'target_shares',
        'delta_shares', 'action'], detailing the required trades.
    """
    # Combine current and target holdings into a single DataFrame
    trade_df = pd.DataFrame({'current': current_shares, 'target': target_shares})
    trade_df = trade_df.fillna(0) # Assume 0 shares for any missing tickers
    
    trade_df['delta_shares'] = (trade_df['target'] - trade_df['current']).astype(int)
    
    # Define the action based on the delta
    trade_df['action'] = 'HOLD'
    trade_df.loc[trade_df['delta_shares'] > 0, 'action'] = 'BUY'
    trade_df.loc[trade_df['delta_shares'] < 0, 'action'] = 'SELL'
    
    # Filter out assets with no required action
    final_trades = trade_df[trade_df['action'] != 'HOLD'].copy()
    
    return final_trades.reset_index().rename(columns={
        'index': 'ticker',
        'current': 'current_shares',
        'target': 'target_shares'
    })


if __name__ == '__main__':
    print("--- Running Trade Calculator Module Standalone Test ---")
    
    # --- GIVEN ---
    current_portfolio = pd.Series({'PETR4.SA': 100, 'VALE3.SA': 50, 'BBDC4.SA': 200})
    target_weights = pd.Series({'PETR4.SA': 0.5, 'VALE3.SA': 0.3, 'ITUB4.SA': 0.2})
    latest_prices = pd.Series({'PETR4.SA': 30.0, 'VALE3.SA': 70.0, 'BBDC4.SA': 15.0, 'ITUB4.SA': 28.0})
    
    # Calculate current portfolio value
    pv = (current_portfolio * latest_prices.reindex(current_portfolio.index)).sum()
    print(f"Current Portfolio Value: R$ {pv:,.2f}")
    
    # --- WHEN ---
    # 1. Calculate target shares
    target_df = calculate_target_shares(target_weights, pv, latest_prices)
    
    # 2. Resolve fractional shares
    target_df = resolve_fractional_shares(target_df, rounding_mode='round')
    
    # 3. Compute the final trade list
    target_shares_series = target_df.set_index('ticker')['target_shares']
    trade_list = compute_trade_list(current_portfolio, target_shares_series)
    
    # --- THEN ---
    print("\nTarget Share Allocation:")
    print(target_df)
    
    print("\n--- Final Trade List ---")
    print(trade_list)
    
    # --- Validation ---
    # BBDC4.SA should be a SELL, as it's not in the target weights.
    assert trade_list[trade_list['ticker'] == 'BBDC4.SA']['action'].iloc[0] == 'SELL'
    assert trade_list[trade_list['ticker'] == 'BBDC4.SA']['delta_shares'].iloc[0] == -200
    
    # ITUB4.SA should be a BUY, as it was not in the original portfolio.
    assert trade_list[trade_list['ticker'] == 'ITUB4.SA']['action'].iloc[0] == 'BUY'
    
    # PETR4.SA's target value should be 50% of PV
    expected_petr4_value = 0.5 * pv
    actual_petr4_value = target_df[target_df['ticker'] == 'PETR4.SA']['target_value'].iloc[0]
    assert np.isclose(expected_petr4_value, actual_petr4_value)
    
    print("\nOK: Trade list calculation is correct and intuitive.")