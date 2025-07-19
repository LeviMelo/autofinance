import pandas as pd
from typing import Dict, List

# Define constants for Brazilian tax rules as per the amendment
TAX_RATE_STOCKS = 0.15
TAX_RATE_BDRS = 0.15
TAX_RATE_ETFS = 0.15
STOCK_EXEMPTION_THRESHOLD = 20000.0  # R$ 20.000,00

def _classify_asset(ticker: str) -> str:
    """Classifies an asset based on its ticker convention."""
    if ticker.endswith("11.SA"):
        return 'ETF'
    elif ticker.endswith("34.SA") or ticker.endswith("35.SA"): # Add other BDR endings if needed
        return 'BDR'
    else: # Default to common stock
        return 'STOCK'

def calculate_monthly_taxes(
    sales_log: List[Dict],
    year: int,
    month: int
) -> Dict:
    """
    Calculates the total capital gains tax due for a specific month.

    This function processes all sales within a given month, applies the relevant
    tax rules (like the R$ 20k exemption), and computes the final tax liability.

    Args:
        sales_log: A list of sale record dictionaries from a TaxLedger.
        year: The year to calculate taxes for.
        month: The month to calculate taxes for.

    Returns:
        A dictionary summarizing the tax calculation for the month.
    """
    start_date = pd.to_datetime(f"{year}-{month}-01")
    end_date = start_date + pd.offsets.MonthEnd(0)
    
    monthly_sales = [
        s for s in sales_log if start_date <= s['sale_date'] <= end_date
    ]
    
    if not monthly_sales:
        return {'tax_due': 0.0, 'details': 'No sales in this month.'}
        
    # --- Step 1: Aggregate sales and gains by asset class ---
    summary = {
        'STOCK': {'gross_sales': 0.0, 'net_gain': 0.0},
        'BDR': {'gross_sales': 0.0, 'net_gain': 0.0},
        'ETF': {'gross_sales': 0.0, 'net_gain': 0.0}
    }
    
    for sale in monthly_sales:
        asset_class = _classify_asset(sale['ticker'])
        if asset_class in summary:
            summary[asset_class]['gross_sales'] += sale['gross_sale_value']
            summary[asset_class]['net_gain'] += sale['capital_gain']
            
    # --- Step 2: Apply Exemption and Tax Rules ---
    tax_due_stock = 0.0
    tax_due_bdr = 0.0
    tax_due_etf = 0.0
    
    # Stocks (Ações)
    stock_summary = summary['STOCK']
    # The exemption applies to the TOTAL gross sales of common stocks in the month.
    if stock_summary['gross_sales'] > STOCK_EXEMPTION_THRESHOLD:
        if stock_summary['net_gain'] > 0:
            tax_due_stock = stock_summary['net_gain'] * TAX_RATE_STOCKS
    
    # BDRs - No exemption
    bdr_summary = summary['BDR']
    if bdr_summary['net_gain'] > 0:
        tax_due_bdr = bdr_summary['net_gain'] * TAX_RATE_BDRS

    # ETFs - No exemption
    etf_summary = summary['ETF']
    if etf_summary['net_gain'] > 0:
        tax_due_etf = etf_summary['net_gain'] * TAX_RATE_ETFS

    # TODO: Implement loss offsetting logic here in a future iteration.
    # For now, we only tax positive net gains in each category.

    total_tax_due = tax_due_stock + tax_due_bdr + tax_due_etf

    return {
        'year': year,
        'month': month,
        'total_tax_due': total_tax_due,
        'details': {
            'STOCK': {**stock_summary, 'tax_due': tax_due_stock, 'exemption_applied': stock_summary['gross_sales'] <= STOCK_EXEMPTION_THRESHOLD},
            'BDR': {**bdr_summary, 'tax_due': tax_due_bdr},
            'ETF': {**etf_summary, 'tax_due': tax_due_etf}
        }
    }


if __name__ == '__main__':
    from .ledger import TaxLedger
    print("--- Running Tax Tracker Module Standalone Test ---")
    
    # --- GIVEN ---
    ledger = TaxLedger()
    # Scenario for March 2023
    
    # Case 1: Stock sale UNDER the exemption limit
    ledger.record_buy('PETR4.SA', 100, 25.0, pd.to_datetime('2023-01-01'))
    ledger.record_sell('PETR4.SA', 100, 30.0, pd.to_datetime('2023-03-05')) # Gross sale = 3000

    # Case 2: Stock sale OVER the exemption limit
    ledger.record_buy('VALE3.SA', 500, 70.0, pd.to_datetime('2023-01-01'))
    ledger.record_sell('VALE3.SA', 500, 80.0, pd.to_datetime('2023-03-10')) # Gross sale = 40000

    # Case 3: BDR sale (no exemption)
    ledger.record_buy('AAPL34.SA', 20, 150.0, pd.to_datetime('2023-01-01'))
    ledger.record_sell('AAPL34.SA', 20, 160.0, pd.to_datetime('2023-03-15')) # Gross sale = 3200

    # Case 4: ETF with a loss
    ledger.record_buy('BOVA11.SA', 10, 110.0, pd.to_datetime('2023-01-01'))
    ledger.record_sell('BOVA11.SA', 10, 105.0, pd.to_datetime('2023-03-20')) # Gross sale = 1050

    # --- WHEN ---
    march_taxes = calculate_monthly_taxes(ledger.sales_log, 2023, 3)

    # --- THEN ---
    print("\n--- Tax Calculation Summary for March 2023 ---")
    import json
    print(json.dumps(march_taxes, indent=2))
    
    # --- Validation ---
    # Total stock gross sales = 3000 (PETR4) + 40000 (VALE3) = 43000.
    # This is > 20k, so the exemption does NOT apply.
    assert not march_taxes['details']['STOCK']['exemption_applied']
    
    # Total stock net gain = (3000-2500) + (40000-35000) = 500 + 5000 = 5500
    expected_stock_tax = 5500 * TAX_RATE_STOCKS
    assert np.isclose(march_taxes['details']['STOCK']['tax_due'], expected_stock_tax)
    
    # BDR gain = (3200 - 3000) = 200
    expected_bdr_tax = 200 * TAX_RATE_BDRS
    assert np.isclose(march_taxes['details']['BDR']['tax_due'], expected_bdr_tax)
    
    # ETF had a loss, so tax should be 0.
    assert march_taxes['details']['ETF']['tax_due'] == 0.0
    
    # Total tax
    expected_total_tax = expected_stock_tax + expected_bdr_tax
    assert np.isclose(march_taxes['total_tax_due'], expected_total_tax)
    
    print(f"\nOK: Total tax due (R$ {march_taxes['total_tax_due']:.2f}) matches expected calculation.")