import pandas as pd
import holidays
from functools import lru_cache

# Using ANBIMA rules from the 'holidays' library as a robust proxy for B3 financial market holidays.
B3_HOLIDAYS_PROVIDER = holidays.Brazil(state="SP", observance_rule=holidays.ANBIMA)


@lru_cache(maxsize=16)
def get_b3_trading_calendar(
    start_date: str | pd.Timestamp, end_date: str | pd.Timestamp
) -> pd.DatetimeIndex:
    """
    Generates a trading calendar for the B3 Exchange for a given date range.

    Args:
        start_date: The start date of the calendar (inclusive).
        end_date: The end date of the calendar (inclusive).

    Returns:
        A pandas DatetimeIndex containing only the trading days for the B3.
        The dates are timezone-naive, representing the trading day.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate business days (Mon-Fri) and then remove specific holidays.
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")

    # The holidays library is faster when a year range is provided.
    years = range(business_days.year.min(), business_days.year.max() + 1)
    b3_holidays_in_range = holidays.Brazil(
        years=years, state="SP", observance_rule=holidays.ANBIMA
    )
    holiday_dates = pd.to_datetime(list(b3_holidays_in_range.keys()))

    trading_calendar = business_days[~business_days.isin(holiday_dates)]
    
    # Ensure timezone-naive as per specification for consistent indexing.
    return trading_calendar.tz_localize(None)


def generate_rebalance_dates(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    frequency: str = "M",
) -> pd.DatetimeIndex:
    """
    Generates a series of rebalancing dates based on a specified frequency.

    This function ensures that rebalance dates are actual B3 trading days.
    For example, for monthly rebalancing, it selects the last trading day of each month.

    Args:
        start_date: The start date for the backtest period.
        end_date: The end date for the backtest period.
        frequency: The rebalancing frequency ('M' for monthly, 'W' for weekly, etc.).
                   Currently, only 'M' is fully supported as per spec.

    Returns:
        A pandas DatetimeIndex of rebalance dates.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    full_calendar = get_b3_trading_calendar(start_date, end_date)

    if frequency.upper() == "M":
        # Get the last day of all months in the period
        month_ends = pd.date_range(start=start_date, end=end_date, freq="M")
        
        # Find the closest preceding trading day for each month-end.
        # The 'asof' method is perfect for this "point-in-time" lookup.
        rebalance_dates = full_calendar.asof(month_ends)
        
        # Remove duplicates (if any) and NaTs (if a month had no trading days before it)
        rebalance_dates = rebalance_dates.dropna().unique()
        return pd.DatetimeIndex(rebalance_dates)

    # Future extension: add weekly, quarterly logic.
    elif frequency.upper() == "W":
        raise NotImplementedError("Weekly rebalancing is not yet implemented.")
    else:
        raise ValueError(f"Unsupported rebalancing frequency: {frequency}")


def apply_publication_lag(
    event_dates: pd.DatetimeIndex | pd.Series,
    lag_days: int,
    calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """
    Shifts event dates forward by a number of trading days to simulate publication lag.

    This is a critical function to prevent lookahead bias when using fundamental data,
    which is published with a delay.

    Args:
        event_dates: A DatetimeIndex or Series of dates when a corporate event
                     (e.g., earnings release) technically occurred.
        lag_days: The number of trading days to wait before the information is
                  considered public knowledge.
        calendar: The full B3 trading calendar for the period.

    Returns:
        A DatetimeIndex with the dates shifted forward to the valid trading day.
    """
    calendar_series = calendar.to_series()
    
    # For each event date, find its location in the trading calendar
    positions = calendar_series.searchsorted(event_dates)
    
    # Shift the positions forward by the lag amount
    # We must ensure we don't go beyond the calendar's boundary
    shifted_positions = positions + lag_days
    shifted_positions = shifted_positions[shifted_positions < len(calendar)]

    return calendar[shifted_positions]


if __name__ == "__main__":
    # --- Example for get_b3_trading_calendar ---
    print("--- Testing get_b3_trading_calendar ---")
    calendar_2024 = get_b3_trading_calendar("2024-01-01", "2024-12-31")
    print(f"Total trading days in 2024: {len(calendar_2024)}")
    carnaval_monday = pd.to_datetime("2024-02-12")
    print(f"Is Feb 12, 2024 (Carnaval) a trading day? {carnaval_monday in calendar_2024}\n")

    # --- Example for generate_rebalance_dates ---
    print("--- Testing generate_rebalance_dates ---")
    rebal_dates = generate_rebalance_dates("2023-01-01", "2023-06-30", frequency="M")
    print("Monthly rebalance dates for H1 2023:")
    print(rebal_dates) # Expect: Jan-31, Feb-28, Mar-31, Apr-28, May-31, Jun-30 (all are trading days)
    
    # Test a month where last day is a weekend
    rebal_dates_sep_2023 = generate_rebalance_dates("2023-09-01", "2023-09-30", "M")
    print("\nRebalance date for Sep 2023 (Sat, Sep 30th is not a trading day):")
    print(rebal_dates_sep_2023) # Expect: 2023-09-29
    
    # --- Example for apply_publication_lag ---
    print("\n--- Testing apply_publication_lag ---")
    full_calendar = get_b3_trading_calendar("2023-03-01", "2023-04-30")
    # Assume fundamentals published for fiscal period ending March 31
    fiscal_period_end = pd.to_datetime(["2023-03-31"])
    print(f"Fiscal period end date: {fiscal_period_end[0].date()}")

    # As per spec, use a 3-day publication lag
    info_becomes_public = apply_publication_lag(fiscal_period_end, lag_days=3, calendar=full_calendar)
    print(f"With a 3-day lag, data is actionable on: {info_becomes_public[0].date()}")
    # March 31, 2023 is a Friday. 3 trading days later should be Wednesday, April 5th.
    # Fri (day 0) -> Mon (day 1) -> Tue (day 2) -> Wed (day 3) -> this is wrong.
    # The lag should be *after* the event.
    # Correct logic: Event on T. Actionable on T+3.
    # March 31 is T. April 3 is T+1, April 4 is T+2, April 5 is T+3. Correct.