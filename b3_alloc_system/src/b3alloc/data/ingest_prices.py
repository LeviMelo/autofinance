# ── standard library ---------------------------------------------------------
from typing import List, Dict
import pandas as pd

# ── third‑party --------------------------------------------------------------
import yfinance as yf
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from yfinance.exceptions import YFRateLimitError  # runtime > 0.2.57

# Optional global throttle (only exists on yfinance 0.2.38‑0.2.58)
try:
    from yfinance.utils import enable_ratelimit  # type: ignore
    enable_ratelimit()
except (ImportError, AttributeError):
    pass  # rely on chunk_size + tenacity back‑off

# ── project‑local ------------------------------------------------------------
from ..utils_dates import get_b3_trading_calendar

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(
        (HTTPError, ConnectionError, TimeoutError, YFRateLimitError)
    ),
)

def _download_chunk_with_tenacity(chunk: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads a single chunk of tickers using yfinance, wrapped in a tenacity retry decorator.
    """
    print(f"  - Downloading chunk: {chunk}")
    df = yf.download(
        tickers=chunk,
        start=start,
        end=end,
        auto_adjust=False,
        actions=True,
        interval="1d",
        threads=False,
        progress=False,
    )
    # yfinance can return an empty DataFrame with a warning for invalid tickers,
    # so we check if all columns are missing data, which can indicate a deeper issue.
    if df.empty or df.columns.empty:
         # Check if it's just a case of no data in the range for valid tickers
        if yf.Ticker(chunk[0]).history(period="1d").empty:
            print(f"Warning: No data for chunk {chunk}, possibly invalid tickers or no history.")
            return pd.DataFrame()
        # Otherwise, raise to trigger retry for transient issues
        raise ConnectionError(f"Download for chunk {chunk} returned an empty frame.")
        
    return df


def fetch_yfinance_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    index_ticker: str = "^BVSP",
    max_retries: int = 5,
    backoff_factor: float = 0.5,
    chunk_size: int = 3,        # was 12
) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical market data for a list of tickers and a benchmark index.

    This function downloads daily Open, High, Low, Close, Adjusted Close, and Volume
    data from Yahoo Finance. It also fetches corporate actions (dividends and splits).
    It includes a retry mechanism with exponential backoff to handle rate limits,
    and downloads tickers in chunks to avoid overwhelming the API.

    Args:
        tickers: A list of B3 ticker symbols to download (e.g., ['PETR4.SA', 'VALE3.SA']).
        start_date: The start date for the data query (YYYY-MM-DD).
        end_date: The end date for the data query (YYYY-MM-DD).
        index_ticker: The ticker for the benchmark index (default: ^BVSP for IBOVESPA).
        max_retries: Maximum number of times to retry the download.
        backoff_factor: Factor to determine the delay between retries (delay = backoff_factor * 2**attempt).
        chunk_size: The number of tickers to download in each batch.

    Returns:
        A dictionary containing:
        - 'prices': A multi-index DataFrame with price and volume data for all tickers.
        - 'index': A DataFrame with price and volume data for the index.
        - 'actions': A dictionary of DataFrames, with each ticker as a key,
                     containing its corporate actions.
    """
    tickers = list(dict.fromkeys(tickers))  # dedupe while preserving order
    print(f"Downloading {len(tickers)} tickers in chunks of {chunk_size}...")

    all_price_frames = []
    failed_chunks = []
    # 1. Download equities in manageable chunks
    for idx in range(0, len(tickers), chunk_size):
        chunk = tickers[idx: idx + chunk_size]
        if not chunk:
            continue
        
        try:
            df_chunk = _download_chunk_with_tenacity(chunk, start=start_date, end=end_date)
            if not df_chunk.empty:
                all_price_frames.append(df_chunk)
        except Exception as e:
            print(f"Chunk {chunk} failed after all retries: {e}")
            failed_chunks.append(chunk)

    # Download the benchmark index separately
    try:
        index_df = _download_chunk_with_tenacity([index_ticker], start=start_date, end=end_date)
    except Exception as e:
        print(f"Index {index_ticker} failed after all retries: {e}")
        index_df = pd.DataFrame()

    # If no equity tickers were requested, just return the index and empty frames
    if not all_price_frames:
        if index_df.empty:
            raise ValueError("Yahoo Finance download failed for both tickers and index.")
        # If ONLY the index was requested and successful, prepare a valid return package
        index_data = index_df.copy()
        index_data.columns = index_data.columns.droplevel(1) # Flatten MultiIndex
        return {"prices": pd.DataFrame(), "index": index_data, "actions": {}}

    # Combine all equity chunks horizontally (columns are MultiIndex)
    data_prices_full = pd.concat(all_price_frames, axis=1)

    # Append the index columns (avoid duplicate column names)
    if not index_df.empty:
        data_prices_full = pd.concat([data_prices_full, index_df], axis=1)

    # Separate index from equities (guard if index failed)
    if index_ticker in data_prices_full.columns.get_level_values(1):
        index_data = data_prices_full.loc[:, (slice(None), index_ticker)].copy()
        index_data.columns = index_data.columns.droplevel(1)
        price_data = data_prices_full.drop(index_ticker, axis=1, level=1)
    else:
        print(f"Warning: Index ticker {index_ticker} missing from download.")
        index_data = pd.DataFrame()
        price_data = data_prices_full

    # Separate prices from actions
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    actions_cols = ['Dividends', 'Stock Splits']
    
    prices = {}
    actions_raw = {}

    for col in price_cols:
        sub_cols = [(col, tk) for tk in tickers if (col, tk) in price_data.columns]
        if sub_cols:
            prices[col] = price_data.loc[:, sub_cols]

    for col in actions_cols:
        sub_cols = [(col, tk) for tk in tickers if (col, tk) in price_data.columns]
        if sub_cols:
            actions_raw[col] = price_data.loc[:, sub_cols]

    # Reformat actions into a more usable dictionary
    actions_dict: Dict[str, pd.DataFrame] = {}
    if actions_raw:
       for tk in tickers:
           frames = []
           for fld, df in actions_raw.items():
               col = (fld, tk)
               if col in df.columns:
                   # select the Series, rename to the field (Dividends / Stock Splits)
                   frames.append(df[col].rename(fld))
           if frames:
               merged = pd.concat(frames, axis=1)
               merged = merged[merged.sum(axis=1) != 0]  # keep only non‑zero rows
               if not merged.empty:
                   actions_dict[tk] = merged

    # Rebuild a clean price DataFrame: rows = date, columns = MultiIndex (ticker, field)
    if price_data.empty:
        raise ValueError("No equity price data available after processing.")

    prices_df = price_data.copy()
    prices_df.columns = prices_df.columns.swaplevel(0, 1)
    prices_df = prices_df.sort_index(axis=1)


    return {"prices": prices_df, "index": index_data, "actions": actions_dict}


def create_equity_price_series(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Orchestrates the download and cleaning of equity price data.
    ...
    """
    b3_calendar = get_b3_trading_calendar(start_date, end_date)
    buffered_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    data_bundle = fetch_yfinance_data(tickers, buffered_start, end_date)
    prices_wide = data_bundle['prices']

    # --- SLICE DATA ---
    adj_close = prices_wide.xs('Adj Close', level=1, axis=1)
    volume = prices_wide.xs('Volume', level=1, axis=1)

    # --- ALIGN TO CALENDAR ---
    adj_close_aligned = adj_close.reindex(b3_calendar).ffill()
    volume_aligned = volume.reindex(b3_calendar).fillna(0)

    # --- NAME INDEXES FOR ROBUST CONVERSION ---
    adj_close_aligned.index.name = "date"
    adj_close_aligned.columns.name = "ticker" # Explicitly name columns
    volume_aligned.index.name = "date"
    volume_aligned.columns.name = "ticker" # Explicitly name columns

    # --- WIDE TO LONG (ROBUST METHOD) ---
    long_adj_close = adj_close_aligned.stack().reset_index(name="adj_close")
    long_volume = volume_aligned.stack().reset_index(name="volume")

    # This merge will now work reliably
    final_df = pd.merge(long_adj_close, long_volume, on=['date', 'ticker'])
    
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