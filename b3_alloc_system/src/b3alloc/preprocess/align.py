import pandas as pd

from ..utils_dates import get_b3_trading_calendar, apply_publication_lag

def align_fundamentals_to_prices(
    fundamentals_df: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    publish_lag_days: int
) -> pd.DataFrame:
    """
    Aligns quarterly fundamental data to a daily price calendar.

    This function performs two critical alignment steps:
    1. It calculates the 'actionable_date' for each fundamental report by
       applying a trading day lag to the 'fiscal_period_end' date. This
       simulates the delay in information publication.
    2. It forward-fills this quarterly data to create a daily panel that
       matches the price series, ensuring that at any point in time, we are
       using the most recently available fundamental data.

    Args:
        fundamentals_df: A long-format DataFrame of quarterly fundamentals with
                         columns ['ticker', 'fiscal_period_end', ...].
        price_dates: A DatetimeIndex of all trading days for which price
                     data is available.
        publish_lag_days: The number of trading days to wait after a fiscal
                          period ends before the data is considered public.

    Returns:
        A daily, long-format DataFrame with tickers and dates as the index,
        containing the correctly lagged and forward-filled fundamental data.
    """
    if fundamentals_df.empty:
        raise ValueError("Input fundamentals_df cannot be empty.")
    if price_dates.empty:
        raise ValueError("Input price_dates cannot be empty.")

    # 1. Determine the full trading calendar for the alignment period
    # This must encompass both fundamental and price date ranges.
    cal_start = min(fundamentals_df['fiscal_period_end'].min(), price_dates.min())
    cal_end = max(fundamentals_df['fiscal_period_end'].max(), price_dates.max())
    full_calendar = get_b3_trading_calendar(cal_start, cal_end)
    
    # 2. Calculate the 'actionable_date' for each fundamental report
    # We group by ticker to handle cases where different tickers have different
    # fiscal period end dates.
    
    # The 'publish_date' column is often missing or unreliable from scrapers.
    # The project spec dictates using 'fiscal_period_end' + a fixed lag as the
    # reliable point-in-time marker.
    
    # Drop any existing publish_date column to avoid confusion
    if 'publish_date' in fundamentals_df.columns:
        fundamentals_df = fundamentals_df.drop(columns=['publish_date'])
        
    lagged_df_list = []
    for ticker, group in fundamentals_df.groupby('ticker'):
        group = group.copy()
        event_dates = pd.to_datetime(group['fiscal_period_end'])
        
        # Use the utility function to shift the dates
        group['actionable_date'] = apply_publication_lag(
            event_dates=event_dates,
            lag_days=publish_lag_days,
            calendar=full_calendar
        )
        lagged_df_list.append(group)

    if not lagged_df_list:
        raise ValueError("Could not calculate actionable dates for any tickers.")
        
    lagged_fundamentals = pd.concat(lagged_df_list)
    lagged_fundamentals = lagged_fundamentals.dropna(subset=['actionable_date'])
    
    # 3. Create the daily panel by forward-filling
    # We set a multi-index on the data we want to fill forward.
    lagged_fundamentals = lagged_fundamentals.set_index(['ticker', 'actionable_date'])
    lagged_fundamentals = lagged_fundamentals.drop(columns=['fiscal_period_end'])
    lagged_fundamentals = lagged_fundamentals.sort_index()

    # Create the target daily index grid (all tickers for all price dates)
    tickers = fundamentals_df['ticker'].unique()
    daily_panel_index = pd.MultiIndex.from_product(
        [tickers, price_dates], names=['ticker', 'date']
    )
    
    # Reindex our lagged data onto this daily grid.
    # The `groupby('ticker').ffill()` is the key step. It ensures that we
    # forward-fill data for each ticker independently.
    daily_fundamentals = lagged_fundamentals.reindex(daily_panel_index)
    daily_fundamentals = daily_fundamentals.groupby(level='ticker').ffill()
    
    # Drop any rows that still have NaNs (e.g., assets that IPO'd mid-period)
    daily_fundamentals = daily_fundamentals.dropna()

    print("Successfully aligned fundamental data to daily price calendar.")
    return daily_fundamentals.reset_index()


if __name__ == '__main__':
    print("--- Running Align Module Standalone Test ---")

    # 1. Create dummy input data
    tickers = ['TICKER_A', 'TICKER_B']
    
    # Quarterly fundamentals for 2023
    dummy_fundamentals = pd.DataFrame({
        'ticker': ['TICKER_A', 'TICKER_A', 'TICKER_B', 'TICKER_B'],
        'fiscal_period_end': pd.to_datetime([
            "2022-12-31", "2023-03-31", "2022-12-31", "2023-03-31"
        ]),
        'book_equity': [1000, 1050, 2000, 1950],
        'shares_outstanding': [100, 100, 50, 50]
    })

    # Daily prices for part of 2023
    price_dates = get_b3_trading_calendar("2023-03-01", "2023-04-30")
    
    PUBLISH_LAG_DAYS = 3 # As per spec

    print("--- Input Data ---")
    print("Fundamentals:")
    print(dummy_fundamentals)
    print(f"\nPrice Dates Range: {price_dates.min().date()} to {price_dates.max().date()}")
    print(f"Publication Lag: {PUBLISH_LAG_DAYS} trading days")
    
    # 2. Run the alignment function
    try:
        daily_panel = align_fundamentals_to_prices(
            dummy_fundamentals, price_dates, PUBLISH_LAG_DAYS
        )
        
        print("\n--- Output ---")
        print("Shape of daily fundamental panel:", daily_panel.shape)
        
        # 3. Validation
        print("\n--- Validation ---")
        
        # Check TICKER_A around the Q1-2023 earnings release
        # Fiscal end: 2023-03-31 (Friday)
        # B3 Calendar around that time: Mar-31, Apr-03, Apr-04, Apr-05, Apr-06 (Good Friday), Apr-10
        # Actionable date should be 3 trading days AFTER Mar-31, which is Apr-05.
        
        data_before_release = daily_panel[
            (daily_panel['ticker'] == 'TICKER_A') &
            (daily_panel['date'] == pd.to_datetime('2023-04-04'))
        ]
        
        data_on_release_day = daily_panel[
            (daily_panel['ticker'] == 'TICKER_A') &
            (daily_panel['date'] == pd.to_datetime('2023-04-05'))
        ]

        print("Data available on April 4th (should be from Q4-2022):")
        print(data_before_release[['date', 'book_equity']])
        assert data_before_release['book_equity'].iloc[0] == 1000

        print("\nData available on April 5th (should be from Q1-2023):")
        print(data_on_release_day[['date', 'book_equity']])
        assert data_on_release_day['book_equity'].iloc[0] == 1050
        
        print("\nOK: Lookahead bias prevention is working correctly.")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during testing: {e}")
        traceback.print_exc()