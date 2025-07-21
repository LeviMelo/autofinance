import pandas as pd
import brfinance as brf
import time
from typing import List, Optional
import numpy as np
import functools         # ← add this
import yfinance as yf
import requests
from bs4 import BeautifulSoup

from brfinance import CVMAsyncBackend             # NEW (brfinance ≥ 1.2.0)
import logging

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)   # ADD

yf.enable_debug_mode()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

@functools.lru_cache(maxsize=128)
def _get_cvm_code_from_b3(ticker: str) -> Optional[str]:
   """
   Resolve a B3 ticker (e.g. 'PETR4') to its CVM code via the nightly CSV:
     https://arquivos.b3.com.br/emissores/EmissoresListados.csv
   """
   base = ticker.split(".")[0].upper()
   url = "https://arquivos.b3.com.br/emissores/EmissoresListados.csv"
   try:
       resp = requests.get(url, timeout=30)
       resp.raise_for_status()
       for line in resp.text.splitlines():
           cols = [c.strip() for c in line.split(";")]
           # CSV columns: Ticker;CodCVM;...
           if len(cols) >= 2 and cols[0].upper() == base and cols[1].isdigit():
               return cols[1]
   except Exception as e:
       logging.warning(f"Failed to fetch CVM code for {base} from CSV: {e}")
   logging.warning(f"CVM code not found for {base} via CSV.")
   return None


def _fetch_shares_outstanding_from_cvm(ticker_sa: str) -> Optional[pd.DataFrame]:
    """
    Return a time–series DataFrame indexed by fiscal_period_end with one column:
        shares_outstanding  (float)

    Data source: CVM “shares” canvas, downloaded through brfinance ≥ 1.1.0.
    Falls back to None if the company is missing or the canvas is empty.
    """
    cvm_backend = CVMAsyncBackend()            # brfinance async helper
    cleaned_ticker = ticker_sa.split(".")[0]  # 'PETR4.SA' ➜ 'PETR4'

    try:
        # 1) map B3 ticker → CVM code (one cheap web‑scrape; cached per run)
        cvm_code = _get_cvm_code_from_b3(cleaned_ticker)

        if cvm_code is None:
            logging.warning(f"CVM code not found for {ticker_sa}.")
            return None

        # 2) download & convert to DataFrame
        df_cvm = cvm_backend.get_canvas_df(
            company_code=cvm_code,
            canvas="shares",
            from_year=2010,
        )
        if df_cvm is None or df_cvm.empty:
            logging.warning(f"CVM shares canvas empty for {ticker_sa}.")
            return None

        # 3) normalise column names (they come in Portuguese)
        df_cvm = (
            df_cvm.rename(
                columns={
                    "dataFimExercicioSocial": "fiscal_period_end",
                    "quantidadeAcoes": "shares_outstanding",
                }
            )
            .loc[:, ["fiscal_period_end", "shares_outstanding"]]
            .dropna()
        )

        df_cvm["fiscal_period_end"] = pd.to_datetime(df_cvm["fiscal_period_end"])
        df_cvm["shares_outstanding"] = (
            df_cvm["shares_outstanding"].astype(str).str.replace("[^0-9]", "", regex=True).astype(float)
        )

        # keep the last filing of each period
        df_cvm = df_cvm.drop_duplicates(subset="fiscal_period_end", keep="last")
        return df_cvm.set_index("fiscal_period_end")

    except Exception as e:
        logging.warning(f"CVM fetch error for {ticker_sa}: {e}")
        return None




def _fetch_single_ticker_fundamentals(ticker_sa: str) -> pd.DataFrame:
    """
    Return a quarterly DataFrame with columns:
        fiscal_period_end | book_equity | shares_outstanding | ticker | publish_date | book_per_share
    Uses yfinance for book equity, CVM for shares; degrades gracefully.

    The function never raises – it returns an empty DataFrame if it truly fails.
    """
    tkr = yf.Ticker(ticker_sa)

    try:
        # ---------- 1) BOOK EQUITY ----------
        def _extract_equity(df: pd.DataFrame) -> pd.DataFrame:
            logging.info(f"Columns returned for {ticker_sa}: {list(df.columns)}")

            if df.empty:
                return pd.DataFrame()

            equity_cols = [
                c for c in df.columns
                if any(key in c.lower() for key in ("equity", "patrim", "total stockholder"))
            ]
            if not equity_cols:
                return pd.DataFrame()

            equity_col = equity_cols[0]
            logging.info(f"Using equity column: {equity_col} for {ticker_sa}")

            out = df[[equity_col]].rename(columns={equity_col: "book_equity"})  # ← NO .T here
            out.index.name = "fiscal_period_end"
            return out


        # Try quarterly first, then annual.
        book_df = _extract_equity(tkr.quarterly_balance_sheet.T)
        if book_df.empty:
            book_df = _extract_equity(tkr.balance_sheet.T)

        # Ultimate fallback – single point from the `info` dict
        if book_df.empty:
            equity_now = tkr.info.get("totalStockholderEquity")
            if equity_now is not None:
                book_df = pd.DataFrame(
                    {
                        "fiscal_period_end": [pd.Timestamp.today().normalize()],
                        "book_equity": [equity_now],
                    }
                )
            else:
                logging.warning(f"Book equity not found for {ticker_sa}.")
                return pd.DataFrame()

        # ---------- 2) SHARES OUTSTANDING ----------
        shares_df = _fetch_shares_outstanding_from_cvm(ticker_sa)
        if shares_df is not None and not shares_df.empty:
            fundamentals = pd.merge(
                book_df.reset_index(), shares_df.reset_index(),
                how="left", on="fiscal_period_end"
            )
            fundamentals["shares_outstanding"].ffill(inplace=True)
        else:
            current_sh = tkr.info.get("sharesOutstanding", np.nan)
            fundamentals = book_df.reset_index().assign(shares_outstanding=current_sh)

        # ---------- 3) FINAL FORMATTING ----------
        fundamentals["ticker"] = ticker_sa
        fundamentals["publish_date"] = pd.NaT
        fundamentals["book_per_share"] = (
            fundamentals["book_equity"] / fundamentals["shares_outstanding"]
        )
        fundamentals.dropna(subset=["book_equity", "shares_outstanding"], inplace=True)
        return fundamentals

    except Exception as e:
        logging.warning(f"Fundamentals fetch failed for {ticker_sa}: {e}")
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