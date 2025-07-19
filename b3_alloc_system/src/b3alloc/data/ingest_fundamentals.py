import pandas as pd
import requests
import time
from typing import List, Tuple, Dict
import re

# This scraper targets the non-official, publicly available API used by
# the StatusInvest website. It is subject to change without notice.
# Using a browser-like User-Agent is essential to avoid being blocked.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "https://statusinvest.com.br"


def _clean_ticker_and_get_type(ticker_sa: str) -> Tuple[str, str]:
    """
    Cleans a '.SA' ticker and determines its type (stock or bdr) for the API endpoint.
    
    Example:
    'PETR4.SA' -> ('petr4', 'acao')
    'AAPL34.SA' -> ('aapl34', 'bdr')
    """
    # Remove .SA suffix and convert to lowercase
    cleaned = ticker_sa.split('.')[0].lower()
    
    # Use regex to find the number in the ticker
    match = re.search(r'(\d+)$', cleaned)
    if match:
        num = int(match.group(1))
        # BDRs typically end in 31, 32, 33, 34, 35, 39
        if num >= 31 and num <= 39:
            return cleaned, 'bdr'
            
    return cleaned, 'acao'


def _fetch_single_ticker_fundamentals(ticker_sa: str) -> pd.DataFrame:
    """
    Fetches historical fundamentals for a single ticker from StatusInvest's API.
    """
    cleaned_ticker, asset_type = _clean_ticker_and_get_type(ticker_sa)
    
    # Construct the correct API endpoint based on asset type
    # This corresponds to the XHR request made by the website's frontend.
    api_url = f"{BASE_URL}/{asset_type}/getbs?companyName={cleaned_ticker}&type=1"
    
    response = requests.get(api_url, headers=HEADERS)
    response.raise_for_status()
    
    json_data = response.json()
    if not json_data.get('success') or not json_data.get('data'):
        # print(f"Warning: No fundamental data found for {ticker_sa}")
        return pd.DataFrame()

    # The data is in a list of dictionaries, one for each fiscal period
    df = pd.DataFrame(json_data['data'])
    
    # Rename columns to match our canonical schema
    df = df.rename(columns={
        'patrimonioLiquido': 'book_equity',
        'vpa': 'book_per_share',
        'data': 'fiscal_period_end'
    })

    # Select only the columns we need
    required_cols = ['fiscal_period_end', 'book_equity', 'book_per_share']
    df = df[required_cols]
    
    # Data Cleaning and Type Conversion
    df['fiscal_period_end'] = pd.to_datetime(df['fiscal_period_end'], format='%d/%m/%Y')
    
    for col in ['book_equity', 'book_per_share']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # As per spec, calculate shares outstanding from book equity and book per share
    # shares = book_equity / book_per_share
    # Add a small epsilon to avoid division by zero
    df['shares_outstanding'] = df['book_equity'] / (df['book_per_share'] + 1e-9)
    df['shares_outstanding'] = df['shares_outstanding'].round(0)
    
    # The API does not provide a 'publish_date'. This will be handled by applying a
    # fixed lag in a later processing step, as per the project specification.
    df['publish_date'] = pd.NaT # Explicitly state that it's not available from source
    
    df['ticker'] = ticker_sa
    
    return df


def create_fundamentals_series(
    tickers: List[str]
) -> pd.DataFrame:
    """
    Orchestrates downloading and compiling quarterly fundamentals for a list of tickers.

    Args:
        tickers: A list of B3 ticker symbols ending in .SA.

    Returns:
        A long-format DataFrame with the canonical fundamentals schema.
    """
    all_fundamentals = []
    print(f"Fetching fundamentals for {len(tickers)} tickers from StatusInvest...")
    
    for i, ticker in enumerate(tickers):
        try:
            # Polite scraping: pause between requests to avoid overloading the server
            time.sleep(0.2)
            
            print(f"({i+1}/{len(tickers)}) Fetching {ticker}...")
            
            ticker_df = _fetch_single_ticker_fundamentals(ticker)
            if not ticker_df.empty:
                all_fundamentals.append(ticker_df)
        except requests.exceptions.HTTPError as e:
            print(f"  -> Failed to fetch {ticker}: {e.response.status_code}")
        except Exception as e:
            print(f"  -> An unexpected error occurred for {ticker}: {e}")

    if not all_fundamentals:
        raise ValueError("Failed to retrieve fundamental data for all tickers.")
        
    final_df = pd.concat(all_fundamentals, ignore_index=True)
    
    # Final column ordering to match schema
    final_df = final_df[[
        'fiscal_period_end',
        'publish_date',
        'ticker',
        'shares_outstanding',
        'book_equity',
        'book_per_share'
    ]]
    
    print("\nSuccessfully created fundamentals series.")
    return final_df.sort_values(by=['ticker', 'fiscal_period_end']).reset_index(drop=True)


if __name__ == '__main__':
    # Standalone test for the module
    TEST_TICKERS = ["PETR4.SA", "WEGE3.SA", "MGLU3.SA", "AAPL34.SA", "NON_EXISTENT.SA"]
    
    print("--- Running Fundamentals Ingestion Module Standalone Test ---")

    try:
        fundamentals_df = create_fundamentals_series(TEST_TICKERS)
        
        print("\n--- Results ---")
        print("Shape of the final DataFrame:", fundamentals_df.shape)
        
        print("\nLatest data for PETR4.SA:")
        print(fundamentals_df[fundamentals_df['ticker'] == 'PETR4.SA'].tail(1))
        
        print("\nLatest data for AAPL34.SA (BDR):")
        print(fundamentals_df[fundamentals_df['ticker'] == 'AAPL34.SA'].tail(1))

        # Check for any missing values, which would be an error
        # (except in publish_date, which is intentionally null)
        cols_to_check = ['fiscal_period_end', 'ticker', 'shares_outstanding', 'book_equity', 'book_per_share']
        if fundamentals_df[cols_to_check].isnull().values.any():
            print("\nERROR: Null values found in critical fundamental data!")
            print(fundamentals_df[fundamentals_df[cols_to_check].isnull().any(axis=1)])
        else:
            print("\nOK: No unexpected null values found.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()