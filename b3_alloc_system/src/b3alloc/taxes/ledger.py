import pandas as pd
from typing import List, Dict, Tuple
import numpy as np

class TaxLedger:
    """
    Manages a detailed, lot-by-lot inventory of asset positions for tax purposes.

    This ledger tracks individual purchase lots (date, shares, price) to enable
    accurate cost-basis tracking using the FIFO (First-In, First-Out) method.
    """

    def __init__(self):
        # A list of dictionaries, where each dict is a "lot"
        self.lots = []
        # A list to record every sale for tax reporting
        self.sales_log = []

    def record_buy(self, ticker: str, shares: int, price: float, date: pd.Timestamp):
        """Records the purchase of an asset lot."""
        if shares <= 0 or price <= 0:
            return
        
        self.lots.append({
            'ticker': ticker,
            'shares_remaining': int(shares),
            'purchase_price': float(price),
            'purchase_date': date,
            'lot_id': f"{date.strftime('%Y%m%d%H%M%S')}-{ticker}-{shares}" # Unique ID
        })
        # Sort by date to ensure FIFO logic works correctly
        self.lots.sort(key=lambda x: x['purchase_date'])

    def record_sell(
        self, ticker: str, shares_to_sell: int, sale_price: float, sale_date: pd.Timestamp
    ) -> float:
        """
        Records a sale and calculates the capital gain using the FIFO method.

        This method iterates through the oldest lots of the specified ticker,
        "consuming" them to fulfill the sale. It calculates the total cost basis
        and logs the sale's details.

        Args:
            ticker: The ticker being sold.
            shares_to_sell: The number of shares to sell.
            sale_price: The price at which the shares are sold.
            sale_date: The date of the sale.

        Returns:
            The total capital gain (positive) or loss (negative) from the sale.
        """
        if shares_to_sell <= 0:
            return 0.0

        relevant_lots = [lot for lot in self.lots if lot['ticker'] == ticker and lot['shares_remaining'] > 0]
        
        if sum(lot['shares_remaining'] for lot in relevant_lots) < shares_to_sell:
            # This would indicate an attempt to sell more shares than owned.
            # In a real system, this should be a critical error.
            print(f"Warning: Attempted to sell {shares_to_sell} of {ticker}, but only {sum(lot['shares_remaining'] for lot in relevant_lots)} owned.")
            # Sell what's available
            shares_to_sell = sum(lot['shares_remaining'] for lot in relevant_lots)

        shares_sold_so_far = 0
        total_cost_basis = 0.0
        
        for lot in relevant_lots:
            if shares_sold_so_far >= shares_to_sell:
                break
            
            shares_from_this_lot = min(shares_to_sell - shares_sold_so_far, lot['shares_remaining'])
            
            # Update the lot
            lot['shares_remaining'] -= shares_from_this_lot
            
            # Accumulate cost basis
            cost_of_this_portion = shares_from_this_lot * lot['purchase_price']
            total_cost_basis += cost_of_this_portion
            
            shares_sold_so_far += shares_from_this_lot

        # Log the sale
        gross_sale_value = shares_to_sell * sale_price
        capital_gain = gross_sale_value - total_cost_basis
        
        self.sales_log.append({
            'sale_date': sale_date,
            'ticker': ticker,
            'shares_sold': shares_to_sell,
            'sale_price': sale_price,
            'gross_sale_value': gross_sale_value,
            'cost_basis': total_cost_basis,
            'capital_gain': capital_gain
        })
        
        return capital_gain

    def get_current_holdings(self) -> pd.Series:
        """Calculates the total number of shares currently held for each ticker."""
        holdings = {}
        for lot in self.lots:
            if lot['shares_remaining'] > 0:
                holdings[lot['ticker']] = holdings.get(lot['ticker'], 0) + lot['shares_remaining']
        return pd.Series(holdings, dtype=int)

if __name__ == '__main__':
    print("--- Running Tax Ledger Module Standalone Test ---")
    
    # --- GIVEN ---
    ledger = TaxLedger()
    
    # 1. A series of buys for PETR4.SA at different prices
    ledger.record_buy(ticker='PETR4.SA', shares=100, price=25.0, date=pd.to_datetime('2023-01-10'))
    ledger.record_buy(ticker='VALE3.SA', shares=50,  price=70.0, date=pd.to_datetime('2023-01-15'))
    ledger.record_buy(ticker='PETR4.SA', shares=200, price=30.0, date=pd.to_datetime('2023-02-20'))
    
    print("--- Initial Holdings ---")
    print(ledger.get_current_holdings())
    assert ledger.get_current_holdings()['PETR4.SA'] == 300
    
    # --- WHEN ---
    # 2. Sell 150 shares of PETR4.SA
    print("\n--- Selling 150 shares of PETR4.SA ---")
    gain1 = ledger.record_sell(ticker='PETR4.SA', shares_to_sell=150, sale_price=35.0, date=pd.to_datetime('2023-03-01'))
    
    # --- THEN ---
    # FIFO Logic:
    # The first 100 shares sold come from the first lot (bought at R$ 25.0).
    # The next 50 shares sold come from the second lot (bought at R$ 30.0).
    # Cost basis = (100 * 25.0) + (50 * 30.0) = 2500 + 1500 = 4000
    # Gross sale = 150 * 35.0 = 5250
    # Expected gain = 5250 - 4000 = 1250
    
    print(f"Calculated Capital Gain: R$ {gain1:.2f}")
    assert np.isclose(gain1, 1250.0), "FIFO capital gain calculation is incorrect."
    
    # Check remaining holdings
    # First lot of PETR4 should be empty.
    # Second lot should have 200 - 50 = 150 shares remaining.
    remaining_holdings = ledger.get_current_holdings()
    print("\n--- Holdings After Sale ---")
    print(remaining_holdings)
    assert remaining_holdings['PETR4.SA'] == 150
    
    petr4_lot1 = next(lot for lot in ledger.lots if lot['purchase_price'] == 25.0)
    assert petr4_lot1['shares_remaining'] == 0
    
    petr4_lot2 = next(lot for lot in ledger.lots if lot['purchase_price'] == 30.0)
    assert petr4_lot2['shares_remaining'] == 150
    
    print("\nOK: Ledger correctly applied FIFO logic and updated lot shares.")

    # --- Test Edge Case: Selling more shares than available ---
    print("\n--- Testing selling more shares than available ---")
    gain2 = ledger.record_sell(ticker='VALE3.SA', shares_to_sell=100, sale_price=80.0, date=pd.to_datetime('2023-03-05'))
    # Expected: Should sell the 50 available shares and log the gain for that amount.
    expected_gain2 = (50 * 80.0) - (50 * 70.0) # (Sale Value) - (Cost Basis)
    assert np.isclose(gain2, expected_gain2), "Gain calculation for partial sell-off is incorrect."
    assert ledger.get_current_holdings().get('VALE3.SA', 0) == 0, "All shares of VALE3.SA should have been sold."
    print("OK: Ledger correctly handled attempt to sell more shares than owned.")