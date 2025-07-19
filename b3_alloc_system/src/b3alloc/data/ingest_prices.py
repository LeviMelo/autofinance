import pandas as pd
import yfinance as yf
from typing import List, Dict

# Assuming this script is run from a context where 'b3alloc' is in the python path
from ..utils_dates import get_b3_trading_calendar

def fetch_yfinance_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    index_ticker: str = "^BVSP",
) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical market data for a list of tickers and a benchmark index.

    This function downloads daily Open, High, Low, Close, Adjusted Close, and Volume
    data from Yahoo Finance. It also fetches corporate actions (dividends and splits).

    Args:
        tickers: A list of B3 ticker symbols to download (e.g., ['PETR4.SA', 'VALE3.SA']).
        start_date: The start date for the data query (YYYY-MM-DD).
        end_date: The end date for the data query (YYYY-MM-DD).
        index_ticker: The ticker for the benchmark index (default: ^BVSP for IBOVESPA).

    Returns:
        A dictionary containing:
        - 'prices': A multi-index DataFrame with price and volume data for all tickers.
        - 'index': A DataFrame with price and volume data for the index.
        - 'actions': A dictionary of DataFrames, with each ticker as a key,
                     containing its corporate actions.
    """
    all_tickers_to_download = tickers + [index_ticker]
    print(f"Downloading data for {len(tickers)} assets and index {index_ticker}...")

    # yfinance is more efficient when downloading all tickers in one call
    data = yf.download(
        all_tickers_to_download,
        start=start_date,
        end=end_date,
        auto_adjust=False,  # We want both Close and Adj Close for audit purposes
        actions=True,       # Fetch dividends and stock splits
        progress=True,
    )

    if data.empty:
        raise ValueError("Yahoo Finance returned no data. Check tickers and date range.")

    # Separate index from equities
    index_data = data.loc[:, (slice(None), index_ticker)].copy()
    index_data.columns = index_data.columns.droplevel(1) # Remove ticker level from columns

    price_data = data.drop(index_ticker, axis=1, level=1)
    
    # Separate prices from actions
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    actions_cols = ['Dividends', 'Stock Splits']
    
    prices = price_data.loc[:, (slice(None), price_cols)]
    actions_raw = price_data.loc[:, (slice(None), actions_cols)]
    
    # Reformat actions into a more usable dictionary
    actions_dict = {}
    for ticker in tickers:
        # Filter for non-zero actions for this ticker
        # The column access can be tricky, ensure it's robust
        if ('Dividends', ticker) in actions_raw.columns:
            ticker_actions = actions_raw.loc[:, (actions_cols, ticker)]
            ticker_actions.columns = ticker_actions.columns.droplevel(1) # Drop ticker level
            ticker_actions = ticker_actions[ticker_actions.sum(axis=1) != 0]
            if not ticker_actions.empty:
                actions_dict[ticker] = ticker_actions

    # For multi-level columns, yfinance might return ('Adj Close', 'Ticker'). We want ('Ticker', 'Adj Close')
    prices.columns = prices.columns.swaplevel(0, 1)
    prices = prices.sort_index(axis=1)

    return {"prices": prices, "index": index_data, "actions": actions_dict}


def create_equity_price_series(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Orchestrates the download and cleaning of equity price data.

    This function aligns the downloaded data to the B3 trading calendar and
    formats it into the canonical 'prices_equity_daily' long format.

    Args:
        tickers: List of B3 ticker symbols.
        start_date: The start date for the series.
        end_date: The end date for the series.

    Returns:
        A long-format DataFrame with columns: ['date', 'ticker', 'adj_close', 'volume'].
    """
    b3_calendar = get_b3_trading_calendar(start_date, end_date)
    
    # Add a buffer to start_date for yfinance to handle edge cases
    buffered_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    data_bundle = fetch_yfinance_data(tickers, buffered_start, end_date)
    prices_wide = data_bundle['prices']

    # We only need Adj Close and Volume for the primary table
    adj_close = prices_wide.loc[:, (slice(None), 'Adj Close')]
    adj_close.columns = adj_close.columns.droplevel(0)

    volume = prices_wide.loc[:, (slice(None), 'Volume')]
    volume.columns = volume.columns.droplevel(0)

    # Align to the official B3 calendar, forward-filling gaps
    adj_close_aligned = adj_close.reindex(b3_calendar).ffill()
    volume_aligned = volume.reindex(b3_calendar).fillna(0) # Fill volume gaps with 0

    # Stack to convert from wide to long format
    long_adj_close = adj_close_aligned.stack().rename('adj_close').reset_index()
    long_adj_close = long_adj_close.rename(columns={'level_1': 'ticker'})

    long_volume = volume_aligned.stack().rename('volume').reset_index()
    long_volume = long_volume.rename(columns={'level_1': 'ticker'})

    # Merge into the final canonical format
    final_df = pd.merge(long_adj_close, long_volume, on=['date', 'ticker'])
    
    # Remove rows where price is NaN (e.g., assets that did not exist at start of window)
    final_df = final_df.dropna(subset=['adj_close'])
    
    print(f"Successfully created equity price series for {len(tickers)} tickers.")
    return final_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)


def create_index_series(
    start_date: str, end_date: str, index_ticker: str = "^BVSP"
) -> pd.DataFrame:
    """
    Creates the cleaned, aligned daily series for the benchmark index.

    Args:
        start_date: The start date for the series.
        end_date: The end date for the series.
        index_ticker: The Yahoo Finance ticker for the index.

    Returns:
        A DataFrame indexed by date with 'adj_close' and 'volume'.
    """
    b3_calendar = get_b3_trading_calendar(start_date, end_date)
    buffered_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    
    data_bundle = fetch_yfinance_data([], buffered_start, end_date, index_ticker)
    index_data = data_bundle['index'][['Adj Close', 'Volume']]
    index_data = index_data.rename(columns={'Adj Close': 'adj_close'})

    # Align to calendar and forward fill
    index_aligned = index_data.reindex(b3_calendar).ffill()
    index_aligned = index_aligned.dropna()
    
    print(f"Successfully created index series for {index_ticker}.")
    return index_aligned


if __name__ == '__main__':
    # Standalone test for the module
    TEST_TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA"]
    TEST_START = "2023-01-01"
    TEST_END = "2023-12-31"

    print("--- Running Price Ingestion Module Standalone Test ---")

    try:
        # Test equity price series creation
        print("\n--- Testing Equity Prices ---")
        equity_prices = create_equity_price_series(TEST_TICKERS, TEST_START, TEST_END)
        print("Equity prices shape:", equity_prices.shape)
        print("Data for one ticker (PETR4.SA):")
        print(equity_prices[equity_prices['ticker'] == 'PETR4.SA'].head())
        
        # Verify no missing data for a specific ticker within its lifetime
        petr4_data = equity_prices[equity_prices['ticker'] == 'PETR4.SA']
        if petr4_data['adj_close'].isnull().any():
             print("\nERROR: Found nulls in PETR4.SA data!")
        else:
             print("\nOK: No nulls found in PETR4.SA data.")

        # Test index series creation
        print("\n--- Testing Index Prices ---")
        ibov_prices = create_index_series(TEST_START, TEST_END)
        print("IBOV prices shape:", ibov_prices.shape)
        print(ibov_prices.head())

        if ibov_prices['adj_close'].isnull().any():
             print("\nERROR: Found nulls in IBOV data!")
        else:
             print("\nOK: No nulls found in IBOV data.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()