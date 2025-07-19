import pandas as pd
import investpy
import time
from typing import List
import numpy as np

def _fetch_single_ticker_fundamentals(ticker_sa: str) -> pd.DataFrame:
    """
    Fetches historical fundamentals for a single ticker from invest.com.
    - Book Equity is from the quarterly balance sheet.
    - Shares Outstanding is the latest available figure, used as a proxy for all periods.
    """
    cleaned_ticker = ticker_sa.split('.')[0]
    
    try:
        # 1. Fetch Balance Sheet for Book Equity
        bs_df = investpy.get_stock_financial_summary(
            stock=cleaned_ticker,
            country='brazil',
            summary_type='balance_sheet',
            period='quarterly'
        )
        # Rename and select the 'Total Equity' column
        bs_df = bs_df.rename(columns={'Total Equity': 'book_equity'})
        if 'book_equity' not in bs_df.columns:
            print(f"Warning: 'Total Equity' not found for {ticker_sa}.")
            return pd.DataFrame()
        
        fundamentals_df = bs_df[['book_equity']].copy()

        # 2. Fetch latest Shares Outstanding from general information
        info_df = investpy.get_stock_information(
            stock=cleaned_ticker,
            country='brazil',
            as_json=True
        )
        shares_outstanding = info_df.get('Shares Outstanding')
        if shares_outstanding is None:
            print(f"Warning: Could not retrieve 'Shares Outstanding' for {ticker_sa}.")
            fundamentals_df['shares_outstanding'] = np.nan
        else:
            # The value is a string like "4443236000.0", convert to float
            fundamentals_df['shares_outstanding'] = float(shares_outstanding)

        # 3. Format the DataFrame
        fundamentals_df['ticker'] = ticker_sa
        fundamentals_df = fundamentals_df.reset_index().rename(columns={'Date': 'fiscal_period_end'})
        
        # publish_date is not available, so we will leave it out for now.
        # It can be proxied by adding 3 months to fiscal_period_end if needed later.
        fundamentals_df['publish_date'] = pd.NaT 

        # book_per_share can be calculated later if needed (book_equity / shares_outstanding)
        fundamentals_df['book_per_share'] = np.nan
        
        return fundamentals_df

    except Exception as e:
        print(f"Could not fetch fundamentals for {ticker_sa}: {e}")
        return pd.DataFrame()


def create_fundamentals_series(
    tickers: List[str]
) -> pd.DataFrame:
    """
    Orchestrates downloading and compiling quarterly fundamentals for a list of tickers.
    """
    all_fundamentals = []
    print(f"Fetching fundamentals for {len(tickers)} tickers from invest.com...")
    
    for i, ticker in enumerate(tickers):
        time.sleep(1) # Polite scraping
        print(f"({i+1}/{len(tickers)}) Fetching {ticker}...")
        
        ticker_df = _fetch_single_ticker_fundamentals(ticker)
        if not ticker_df.empty:
            all_fundamentals.append(ticker_df)

    if not all_fundamentals:
        print("Warning: Failed to retrieve fundamental data for any tickers.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_fundamentals, ignore_index=True)
    
    final_df = final_df[[
        'fiscal_period_end',
        'publish_date',
        'ticker',
        'shares_outstanding',
        'book_equity',
        'book_per_share'
    ]]
    
    # Convert types for consistency
    final_df['fiscal_period_end'] = pd.to_datetime(final_df['fiscal_period_end'])
    final_df['publish_date'] = pd.to_datetime(final_df['publish_date'])
    final_df['shares_outstanding'] = pd.to_numeric(final_df['shares_outstanding'])
    final_df['book_equity'] = pd.to_numeric(final_df['book_equity'])

    print("\nSuccessfully created fundamentals series.")
    return final_df.sort_values(by=['ticker', 'fiscal_period_end']).reset_index(drop=True)


if __name__ == '__main__':
    # A standalone test to verify the new logic
    TEST_TICKERS = ["PETR4.SA", "VALE3.SA"]
    
    print("--- Running Fundamentals Ingestion Module Standalone Test ---")
    
    try:
        fundamentals = create_fundamentals_series(TEST_TICKERS)
        if not fundamentals.empty:
            print("\n--- Test Results ---")
            print("Shape:", fundamentals.shape)
            print("Columns:", fundamentals.columns.tolist())
            print("\nData for PETR4.SA:")
            print(fundamentals[fundamentals['ticker'] == 'PETR4.SA'].tail())
            print("\nData for VALE3.SA:")
            print(fundamentals[fundamentals['ticker'] == 'VALE3.SA'].tail())

            # Check for nulls
            if fundamentals['shares_outstanding'].isnull().any():
                print("\nWarning: Found nulls in 'shares_outstanding'.")
            if fundamentals['book_equity'].isnull().any():
                print("\nWarning: Found nulls in 'book_equity'.")
        else:
            print("\nTest failed: No data was returned.")
            
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()