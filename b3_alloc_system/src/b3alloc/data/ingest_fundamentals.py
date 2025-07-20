import pandas as pd
import investpy
import brfinance as brf
import time
from typing import List, Optional
import numpy as np

def _fetch_shares_outstanding_from_cvm(ticker_sa: str) -> Optional[pd.DataFrame]:
    """
    Fetches a time series of historical shares outstanding from CVM (via brfinance).
    """
    cleaned_ticker = ticker_sa.split('.')[0]
    try:
        # brfinance uses the CVM code, which is often the first part of the ticker
        df_cvm = brf.get_canvas(company=cleaned_ticker, from_year=2010, canvas_type="shares")
        if df_cvm is None or df_cvm.empty:
            print(f"Warning: brfinance returned no share data for {ticker_sa}.")
            return None
            
        # Select and rename columns for consistency
        df_cvm = df_cvm[['종료일', '주식수']].rename(columns={
            '종료일': 'fiscal_period_end',
            '주식수': 'shares_outstanding'
        })
        df_cvm['fiscal_period_end'] = pd.to_datetime(df_cvm['fiscal_period_end'])
        # The number can have commas, remove them and convert to numeric
        df_cvm['shares_outstanding'] = df_cvm['shares_outstanding'].astype(str).str.replace(',', '').astype(float)
        
        # Keep only the last report for each quarter to avoid intra-quarter duplicates
        df_cvm = df_cvm.drop_duplicates(subset='fiscal_period_end', keep='last')
        
        return df_cvm.set_index('fiscal_period_end')
        
    except Exception as e:
        print(f"Warning: Could not fetch CVM shares data for {ticker_sa}: {e}")
        return None


def _fetch_single_ticker_fundamentals(ticker_sa: str) -> pd.DataFrame:
    """
    Fetches historical fundamentals for a single ticker.
    - Book Equity is from investpy (quarterly balance sheet).
    - Shares Outstanding is from CVM filings (brfinance) for a proper time series.
      Falls back to investpy's latest figure as a proxy if CVM fails.
    """
    cleaned_ticker = ticker_sa.split('.')[0]
    
    try:
        # 1. Fetch Book Equity from investpy
        bs_df = investpy.get_stock_financial_summary(
            stock=cleaned_ticker,
            country='brazil',
            summary_type='balance_sheet',
            period='quarterly'
        )
        bs_df = bs_df.rename(columns={'Total Equity': 'book_equity'})
        if 'book_equity' not in bs_df.columns:
            print(f"Warning: 'Total Equity' not found for {ticker_sa}.")
            return pd.DataFrame()
        
        bs_df = bs_df[['book_equity']].copy()
        bs_df.index = pd.to_datetime(bs_df.index)
        bs_df.index.name = 'fiscal_period_end'

        # 2. Fetch historical Shares Outstanding from CVM
        shares_df = _fetch_shares_outstanding_from_cvm(ticker_sa)

        # 3. Combine sources or use fallback
        if shares_df is not None:
            # Merge CVM shares with investpy book equity.
            # Use left join to keep all book equity dates, and ffill to fill gaps in share counts.
            fundamentals_df = bs_df.join(shares_df, how='left')
            fundamentals_df['shares_outstanding'] = fundamentals_df['shares_outstanding'].ffill()
        else:
            # Fallback to old method: use latest shares from investpy as proxy
            print(f"Info: Falling back to investpy proxy for shares outstanding for {ticker_sa}.")
            fundamentals_df = bs_df
            info_df = investpy.get_stock_information(stock=cleaned_ticker, country='brazil', as_json=True)
            shares_outstanding_proxy = info_df.get('Shares Outstanding')
            if shares_outstanding_proxy is not None:
                fundamentals_df['shares_outstanding'] = float(shares_outstanding_proxy)
            else:
                fundamentals_df['shares_outstanding'] = np.nan

        # 4. Final Formatting
        fundamentals_df = fundamentals_df.dropna(subset=['book_equity', 'shares_outstanding'])
        fundamentals_df['ticker'] = ticker_sa
        fundamentals_df = fundamentals_df.reset_index()
        
        fundamentals_df['publish_date'] = pd.NaT # Will be calculated later based on lag
        fundamentals_df['book_per_share'] = fundamentals_df['book_equity'] / fundamentals_df['shares_outstanding']
        
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
            print("\nData for PETR4.SA (shows variation in shares):")
            print(fundamentals[fundamentals['ticker'] == 'PETR4.SA'].tail())
            print("\nData for VALE3.SA (shows variation in shares):")
            print(fundamentals[fundamentals['ticker'] == 'VALE3.SA'].tail())

            # Check that BPS is now calculated
            if fundamentals['book_per_share'].isnull().any():
                print("\nWarning: Found nulls in 'book_per_share'.")
        else:
            print("\nTest failed: No data was returned.")
            
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()