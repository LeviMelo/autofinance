import pandas as pd
from typing import Dict, Tuple

class PortfolioLedger:
    """
    Manages the portfolio's state, including holdings, cash, and valuations.

    This class provides a detailed accounting system to track the portfolio's
    composition and value at each time step, serving as the "source of truth"
    for performance measurement and transaction cost analysis.
    """
    def __init__(self, initial_capital: float):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
            
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = pd.Series(dtype=int, name="shares")
        self.history = []
        self.record_state(pd.Timestamp.min) # Record initial state

    def get_portfolio_value(self, current_prices: pd.Series, date: pd.Timestamp) -> float:
        """Calculates the total market value of the portfolio at a given point in time."""
        market_value_of_holdings = (self.holdings * current_prices.reindex(self.holdings.index).fillna(0)).sum()
        return market_value_of_holdings + self.cash

    def record_state(self, date: pd.Timestamp, prices: pd.Series = None):
        """Records the current state (holdings, cash, value) of the portfolio for a given date."""
        if prices is None:
            portfolio_value = self.initial_capital
        else:
            portfolio_value = self.get_portfolio_value(prices, date)
            
        self.history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings.to_dict()
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """Returns the recorded history as a pandas DataFrame."""
        return pd.DataFrame(self.history).set_index('date')

    def execute_trades(
        self,
        trade_list: pd.DataFrame,
        trade_prices: pd.Series,
        date: pd.Timestamp,
        cost_per_trade_bps: int = 0
    ) -> float:
        """
        Updates portfolio holdings and cash by executing a list of trades.

        Args:
            trade_list: A DataFrame with columns ['ticker', 'delta_shares'].
            trade_prices: A Series of prices at which trades are executed.
            date: The timestamp of the trade execution.
            cost_per_trade_bps: Transaction costs in basis points.

        Returns:
            The total transaction cost incurred.
        """
        total_cost = 0.0
        
        for _, trade in trade_list.iterrows():
            ticker = trade['ticker']
            delta = trade['delta_shares']
            price = trade_prices.get(ticker)
            
            if price is None:
                print(f"Warning: No price found for {ticker} on {date.date()}. Skipping trade.")
                continue

            trade_value = delta * price
            trade_cost = abs(trade_value) * (cost_per_trade_bps / 10000.0)
            
            # Update cash
            self.cash -= trade_value  # Cash decreases on BUY, increases on SELL
            self.cash -= trade_cost
            total_cost += trade_cost
            
            # Update holdings
            current_shares = self.holdings.get(ticker, 0)
            new_shares = current_shares + delta
            
            if new_shares > 0:
                self.holdings[ticker] = new_shares
            elif ticker in self.holdings:
                # Remove ticker if we sold all shares
                self.holdings.drop(ticker, inplace=True)
                
            # After each trade, record the new state
            self.record_state(date, trade_prices)
            
        return total_cost


if __name__ == '__main__':
    print("--- Running Portfolio Accounting Module Standalone Test ---")
    
    # --- GIVEN ---
    initial_pv = 100_000.0
    ledger = PortfolioLedger(initial_capital=initial_pv)
    
    start_date = pd.to_datetime("2023-01-02")
    rebal_date = pd.to_datetime("2023-01-03")
    
    prices_t0 = pd.Series({'PETR4.SA': 30.0, 'VALE3.SA': 70.0})
    prices_t1 = pd.Series({'PETR4.SA': 31.0, 'VALE3.SA': 69.0})
    
    # Initial BUY trades
    initial_trades = pd.DataFrame({
        'ticker': ['PETR4.SA', 'VALE3.SA'],
        'delta_shares': [1000, 500] # Buy 1000 PETR4, 500 VALE3
    })
    
    # --- WHEN ---
    # 1. Execute initial trades
    t_costs1 = ledger.execute_trades(initial_trades, prices_t0, start_date, cost_per_trade_bps=10)
    
    # Check state after first set of trades
    state1 = ledger.history[-1]
    value1 = ledger.get_portfolio_value(prices_t0, start_date)
    
    # 2. On the next day, rebalance
    # Sell 200 PETR4, Buy 100 VALE3
    rebal_trades = pd.DataFrame({
        'ticker': ['PETR4.SA', 'VALE3.SA'],
        'delta_shares': [-200, 100]
    })
    t_costs2 = ledger.execute_trades(rebal_trades, prices_t1, rebal_date, cost_per_trade_bps=10)
    
    state2 = ledger.history[-1]
    value2 = ledger.get_portfolio_value(prices_t1, rebal_date)
    
    # --- THEN ---
    print("\n--- State after initial trades ---")
    print(f"Transaction Costs: R$ {t_costs1:.2f}")
    print(f"Holdings: {state1['holdings']}")
    print(f"Cash: R$ {state1['cash']:,.2f}")
    print(f"Portfolio Value: R$ {value1:,.2f}")
    
    # Validation 1
    expected_petr4_value = 1000 * 30.0
    expected_vale3_value = 500 * 70.0
    expected_total_value = expected_petr4_value + expected_vale3_value
    expected_costs = expected_total_value * 0.0010
    assert np.isclose(t_costs1, expected_costs)
    assert np.isclose(state1['cash'], initial_pv - expected_total_value - expected_costs)
    
    print("\n--- State after rebalance ---")
    print(f"Transaction Costs: R$ {t_costs2:.2f}")
    print(f"Holdings: {state2['holdings']}")
    print(f"Cash: R$ {state2['cash']:,.2f}")
    print(f"Portfolio Value: R$ {value2:,.2f}")
    
    # Validation 2
    # New holdings should be 800 PETR4, 600 VALE3
    assert state2['holdings']['PETR4.SA'] == 800
    assert state2['holdings']['VALE3.SA'] == 600
    # Check portfolio value
    expected_value2 = (800 * 31.0) + (600 * 69.0) + state2['cash']
    assert np.isclose(value2, expected_value2)
    
    print("\nOK: Portfolio ledger correctly tracks holdings, cash, and value across trades.")

    # --- Test history recording ---
    print("\n--- Testing history recording ---")
    assert len(ledger.history) == 5, "Expected 5 history records (initial + 2 buys + 2 rebal trades)."
    print("OK: History is being recorded for each transaction.")