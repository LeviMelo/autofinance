# This file marks the 'b3alloc' directory as a Python package.
# It can also be used to define package-level variables or import key functions.

from .ingest_prices import create_equity_price_series, create_index_series
from .ingest_fundamentals import create_fundamentals_series
from .ingest_selic import create_risk_free_series
from .ingest_fx import create_fx_series

__version__ = "0.1.0"