import pandas as pd
import requests
import numpy as np

from ..utils_dates import get_b3_trading_calendar

# The BCB SGS series for USD/BRL closing exchange rate (PTAX) is 1.
BCB_API_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"

def fetch_fx_from_bcb(
    series_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Fetches a time series from the BCB's SGS API. Specifically for FX rates.

    Args:
        series_id: The numerical ID of the SGS series (e.g., 1 for USD/BRL).
        start_date: The start date for the data query.
        end_date: The end date for the data query.

    Returns:
        A pandas DataFrame with the raw data from the API.
    """
    url = BCB_API_BASE_URL.format(series_id=series_id)
    params = {
        "formato": "json",
        "dataInicial": start_date.strftime("%d/%m/%Y"),
        "dataFinal": end_date.strftime("%d/%m/%Y"),
    }
    
    print(f"Fetching FX series {series_id} from {start_date.date()} to {end_date.date()}...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    if not data:
        raise ValueError(f"No FX data returned from BCB API for series {series_id}.")

    df = pd.DataFrame(data)
    return df

def create_fx_series(
    start_date: str, end_date: str, series_id: int = 1, fx_pair: str = "USD_BRL"
) -> pd.DataFrame:
    """
    Constructs the daily FX rate series and its log returns.

    Args:
        start_date: The start date for the series.
        end_date: The end date for the series.
        series_id: The BCB SGS series ID for the FX rate.
        fx_pair: A string name for the FX pair column.

    Returns:
        A pandas DataFrame indexed by trading day, with columns for the
        spot rate and its daily log return.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    # 1. Fetch raw data from BCB
    raw_fx_df = fetch_fx_from_bcb(series_id, start_ts, end_ts)

    # 2. Basic processing
    raw_fx_df["date"] = pd.to_datetime(raw_fx_df["data"], format="%d/%m/%Y")
    raw_fx_df[fx_pair] = pd.to_numeric(raw_fx_df["valor"], errors="coerce")
    
    raw_fx_df = raw_fx_df.set_index("date")[[fx_pair]].sort_index()
    
    # 3. Get the canonical trading calendar
    b3_calendar = get_b3_trading_calendar(start_date, end_date)

    # 4. Align FX data to the trading calendar and forward-fill
    aligned_fx = raw_fx_df.reindex(b3_calendar, method="ffill")
    aligned_fx = aligned_fx.dropna()
    
    # 5. Calculate daily log returns for use as a risk factor
    log_return_col = f"{fx_pair}_log_return"
    # A more direct and efficient way to calculate log returns
    aligned_fx[log_return_col] = np.log(aligned_fx[fx_pair] / aligned_fx[fx_pair].shift(1))
    
    aligned_fx = aligned_fx.dropna() # First row will be NaN
    
    print(f"Successfully created FX series with {len(aligned_fx)} trading days.")
    return aligned_fx

if __name__ == "__main__":
    print("--- Running FX Ingest Module Standalone Test ---")
    
    TEST_START_DATE = "2023-01-01"
    TEST_END_DATE = "2023-06-30"
    
    try:
        fx_df = create_fx_series(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            series_id=1,
            fx_pair="USD_BRL"
        )
        
        print("\n--- Results ---")
        print("Shape of the final DataFrame:", fx_df.shape)
        print("\nFirst 5 rows:")
        print(fx_df.head())
        print("\nLast 5 rows:")
        print(fx_df.tail())
        
        # Check for any missing values
        if fx_df.isnull().values.any():
            print("\nERROR: Missing values found in the final series!")
        else:
            print("\nOK: No missing values found.")
            
        # Validate return calculation
        price_t1 = fx_df['USD_BRL'].iloc[1]
        price_t0 = fx_df['USD_BRL'].iloc[0]
        expected_ret = np.log(price_t1 / price_t0)
        
        actual_ret = fx_df['USD_BRL_log_return'].iloc[0]
        
        assert np.isclose(expected_ret, actual_ret)
        print("\nOK: Log return calculation is validated.")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()