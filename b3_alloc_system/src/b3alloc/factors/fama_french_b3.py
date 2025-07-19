import pandas as pd
import numpy as np
from typing import Dict

def _get_factor_portfolio_assignments(
    data_for_date: pd.DataFrame
) -> pd.Series:
    """
    Assigns each stock to one of six portfolios based on size and B/M ratio.

    This function is called for a single date (the formation date) and uses
    the cross-section of all stocks at that point in time.

    Args:
        data_for_date: A DataFrame with 'market_cap' and 'book_to_market'
                       for all tickers on a single formation date.

    Returns:
        A pandas Series with tickers as index and portfolio assignment
        (e.g., 'Small_High') as values.
    """
    # Filter out ineligible stocks for factor construction, as per spec
    eligible = data_for_date[
        (data_for_date['book_to_market'].notna()) &
        (data_for_date['book_to_market'] > 0) & # Positive book equity
        (data_for_date['market_cap'] > 0)
    ].copy()

    if eligible.empty:
        return pd.Series(dtype=str)

    # 1. Size Breakpoint (SMB): Median market cap
    size_breakpoint = eligible['market_cap'].median()
    eligible['size_bucket'] = np.where(eligible['market_cap'] >= size_breakpoint, 'Big', 'Small')

    # 2. Value Breakpoints (HML): 30th and 70th percentiles of B/M
    bm_p30, bm_p70 = eligible['book_to_market'].quantile([0.3, 0.7])
    eligible['value_bucket'] = 'Mid'
    eligible.loc[eligible['book_to_market'] <= bm_p30, 'value_bucket'] = 'Low' # Growth
    eligible.loc[eligible['book_to_market'] >= bm_p70, 'value_bucket'] = 'High' # Value

    # 3. Combine to form the six portfolios
    assignments = eligible['size_bucket'] + '_' + eligible['value_bucket']
    return assignments


def build_fama_french_factors(
    daily_fundamentals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    market_excess_returns: pd.Series
) -> pd.DataFrame:
    """
    Constructs the daily Fama-French 3 factors (MKT, SMB, HML) for the B3 market.

    Args:
        daily_fundamentals_df: Long-format DataFrame from the align module.
        prices_df: Wide-format DataFrame of daily prices.
        returns_df: Wide-format DataFrame of daily simple returns.
        market_excess_returns: Series of daily IBOV excess returns.

    Returns:
        A DataFrame with a daily time series of the Mkt_excess, SMB, and HML factors.
    """
    print("Building Fama-French 3 factors for B3...")

    # 1. Create a daily panel with all necessary metrics
    # Pivot fundamentals to wide for easier alignment
    shares_wide = daily_fundamentals_df.pivot(index='date', columns='ticker', values='shares_outstanding')
    book_equity_wide = daily_fundamentals_df.pivot(index='date', columns='ticker', values='book_equity')

    # Align all dataframes to the returns index
    prices, returns, shares, book_equity = pd.align(
        prices_df, returns_df, shares_wide, book_equity_wide, join='inner', axis=0
    )
    
    market_cap = prices * shares
    book_to_market = book_equity / market_cap
    
    # 2. Determine portfolio assignments at the beginning of each month
    # This is a common simplification of the quarterly re-formation rule
    formation_dates = returns.resample('M').first().index
    
    all_assignments = pd.DataFrame(index=returns.index, columns=returns.columns)

    print("Determining monthly portfolio assignments...")
    for date in formation_dates:
        if date not in market_cap.index: continue
            
        data_for_formation = pd.DataFrame({
            'market_cap': market_cap.loc[date],
            'book_to_market': book_to_market.loc[date]
        })
        assignments = _get_factor_portfolio_assignments(data_for_formation)
        all_assignments.loc[date] = assignments
    
    # Forward-fill the assignments for the rest of each month
    all_assignments = all_assignments.ffill()

    # 3. Calculate daily returns for the 6 value-weighted portfolios
    portfolio_returns = pd.DataFrame(index=returns.index)
    
    # Previous day's market cap is used for weighting today's returns
    mkt_cap_lagged = market_cap.shift(1)

    print("Calculating daily value-weighted portfolio returns...")
    for portfolio_name in ['Small_Low', 'Small_Mid', 'Small_High', 'Big_Low', 'Big_Mid', 'Big_High']:
        # Create a boolean mask for stocks in this portfolio on each day
        mask = (all_assignments == portfolio_name)
        
        # Calculate weights for each stock within the portfolio
        portfolio_mkt_cap = (mkt_cap_lagged * mask).sum(axis=1)
        weights = (mkt_cap_lagged * mask).div(portfolio_mkt_cap, axis=0).fillna(0)
        
        # Portfolio return is the sum of weighted constituent returns
        portfolio_returns[portfolio_name] = (returns * weights).sum(axis=1)

    # 4. Calculate SMB and HML factor returns
    # SMB = (Small/Low + Small/Mid + Small/High)/3 - (Big/Low + Big/Mid + Big/High)/3
    smb = (
        portfolio_returns[['Small_Low', 'Small_Mid', 'Small_High']].mean(axis=1) -
        portfolio_returns[['Big_Low', 'Big_Mid', 'Big_High']].mean(axis=1)
    )
    
    # HML = (Small/High + Big/High)/2 - (Small/Low + Big/Low)/2
    hml = (
        portfolio_returns[['Small_High', 'Big_High']].mean(axis=1) -
        portfolio_returns[['Small_Low', 'Big_Low']].mean(axis=1)
    )

    # 5. Combine into the final factor panel
    factor_panel = pd.DataFrame({
        'mkt_excess': market_excess_returns,
        'smb': smb,
        'hml': hml
    }).dropna()
    
    print("Successfully built Fama-French factor panel.")
    return factor_panel


if __name__ == '__main__':
    print("--- Running Fama-French Module Standalone Test ---")

    # 1. Create dummy input data for a 2-month period
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="B")
    tickers = ['S_L', 'S_H', 'B_L', 'B_H'] # Small/Low, Small/High, Big/Low, Big/High
    
    # Setup characteristics
    shares = pd.DataFrame({'S_L': 1e6, 'S_H': 1e6, 'B_L': 5e6, 'B_H': 5e6}, index=dates)
    prices = pd.DataFrame({
        'S_L': np.linspace(10, 11, len(dates)), 'S_H': np.linspace(12, 13, len(dates)),
        'B_L': np.linspace(50, 52, len(dates)), 'B_H': np.linspace(60, 61, len(dates)),
    }, index=dates)
    book_equity = pd.DataFrame({ # Low B/M for _L, High B/M for _H
        'S_L': 1e6, 'S_H': 10e6, 'B_L': 20e6, 'B_H': 300e6
    }, index=dates)

    daily_fundamentals = pd.concat([
        shares.stack().rename('shares_outstanding'),
        book_equity.stack().rename('book_equity')
    ], axis=1).reset_index().rename(columns={'level_0':'date', 'level_1':'ticker'})
    
    returns = prices.pct_change().dropna()
    mkt_excess = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
    
    # Align indices for the test
    prices, mkt_excess = prices.align(returns, join='right', axis=0)[0], mkt_excess.align(returns, join='right', axis=0)[0]
    daily_fundamentals = daily_fundamentals[daily_fundamentals.date.isin(returns.index)]

    # 2. Run the builder
    try:
        factor_panel = build_fama_french_factors(daily_fundamentals, prices, returns, mkt_excess)
        
        print("\n--- Output Factor Panel ---")
        print(factor_panel.head())
        print("\n--- Diagnostics ---")
        print(factor_panel.describe())
        
        # 3. Validation
        # On a day where small caps outperform big caps, SMB should be positive.
        # Let's engineer a return for one day
        test_date = pd.to_datetime('2023-02-15')
        if test_date in returns.index:
            returns.loc[test_date, ['S_L', 'S_H']] = 0.02  # Small caps jump 2%
            returns.loc[test_date, ['B_L', 'B_H']] = -0.01 # Big caps drop 1%
            
            # Re-run with the specific return shock
            factor_panel_shock = build_fama_french_factors(daily_fundamentals, prices, returns, mkt_excess)
            smb_on_shock_day = factor_panel_shock.loc[test_date, 'smb']
            
            print(f"\nSMB on shock day ({test_date.date()}): {smb_on_shock_day:.4f}")
            assert smb_on_shock_day > 0, "SMB should be positive when small caps outperform."
            print("OK: SMB sign responds correctly to return shocks.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()