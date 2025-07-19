import pandas as pd
import requests
from io import StringIO
from typing import Dict, Any

# Assuming this script is run from a context where 'b3alloc' is in the python path
from ..utils_dates import get_b3_trading_calendar, B3_HOLIDAYS_PROVIDER
from ..config import Config

# The specification refers to BCB SGS series 11 for SELIC.
# API documentation: https://dadosabertos.bcb.gov.br/dataset/11-taxa-de-juros---selic/resource/71bcb420-5503-4a72-b651-70a273574b41
BCB_API_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
TRADING_DAYS_PER_YEAR = 252 # As per specification

def fetch_selic_from_bcb(
    series_id: int, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Fetches a time series from the BCB's SGS API.

    Args:
        series_id: The numerical ID of the SGS series (e.g., 11 for SELIC).
        start_date: The start date for the data query.
        end_date: The end date for the data query.

    Returns:
        A pandas DataFrame with the raw data from the API.

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
    """
    url = BCB_API_BASE_URL.format(series_id=series_id)
    params = {
        "formato": "json",
        "dataInicial": start_date.strftime("%d/%m/%Y"),
        "dataFinal": end_date.strftime("%d/%m/%Y"),
    }
    
    print(f"Fetching SELIC series {series_id} from {start_date.date()} to {end_date.date()}...")
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    
    data = response.json()
    if not data:
        raise ValueError(f"No data returned from BCB API for series {series_id} in the given date range.")

    df = pd.DataFrame(data)
    return df


def create_risk_free_series(
    start_date: str, end_date: str, series_id: int = 11
) -> pd.DataFrame:
    """
    Constructs the daily risk-free rate series for the B3 market.

    This function orchestrates fetching the SELIC rate, processing it, and
    aligning it with the official B3 trading calendar.

    Args:
        start_date: The start date for the series (e.g., '2010-01-01').
        end_date: The end date for the series (e.g., '2025-01-01').
        series_id: The BCB SGS series ID for the risk-free rate.
                   Defaults to 11 (SELIC).

    Returns:
        A pandas DataFrame indexed by trading day, with columns for the
        daily risk-free rate and its annualized equivalent.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    # 1. Fetch raw data from BCB
    raw_selic_df = fetch_selic_from_bcb(series_id, start_ts, end_ts)

    # 2. Basic processing
    # The 'valor' is the daily rate as a percentage. Convert to decimal.
    raw_selic_df["date"] = pd.to_datetime(raw_selic_df["data"], format="%d/%m/%Y")
    raw_selic_df["rf_daily"] = pd.to_numeric(raw_selic_df["valor"], errors="coerce") / 100.0
    
    # Set date as index for alignment
    raw_selic_df = raw_selic_df.set_index("date")[["rf_daily"]]
    raw_selic_df = raw_selic_df.sort_index()
    
    # 3. Get the canonical trading calendar
    b3_calendar = get_b3_trading_calendar(start_date, end_date)

    # 4. Align SELIC data to the trading calendar
    # Reindex with the trading calendar and forward-fill missing values
    # (e.g., SELIC rate from Friday applies to the following Monday if no change)
    aligned_rf = raw_selic_df.reindex(b3_calendar, method="ffill")
    
    # Drop any initial NaNs if the backtest starts before the SELIC data
    aligned_rf = aligned_rf.dropna()

    # 5. Calculate annualized rate for diagnostics and reporting
    aligned_rf["selic_annualized"] = (
        (1 + aligned_rf["rf_daily"]) ** TRADING_DAYS_PER_YEAR
    ) - 1
    
    aligned_rf.index.name = "date"
    
    print(f"Successfully created risk-free series with {len(aligned_rf)} trading days.")
    return aligned_rf


if __name__ == "__main__":
    # Example of how to run this module directly for testing
    # This requires a dummy config or hardcoded dates
    
    TEST_START_DATE = "2022-01-01"
    TEST_END_DATE = "2022-12-31"
    
    print("--- Running SELIC Ingestion Module Standalone Test ---")
    
    try:
        risk_free_df = create_risk_free_series(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            series_id=11
        )
        
        print("\n--- Results ---")
        print("Shape of the final DataFrame:", risk_free_df.shape)
        print("\nFirst 5 rows:")
        print(risk_free_df.head())
        print("\nLast 5 rows:")
        print(risk_free_df.tail())
        
        print("\nBasic Stats:")
        print(risk_free_df.describe())
        
        # Check for any missing values, which would be an error
        if risk_free_df.isnull().values.any():
            print("\nERROR: Missing values found in the final series!")
        else:
            print("\nOK: No missing values found in the final series.")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")