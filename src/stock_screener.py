"""
Defines preprocessing functions for stock selection based on the Kelly criterion, 
average daily range, volatility, and positive returns.

"""

import pandas as pd

from src.utils import (
    empirical_annualized_volatility,
    kelly_stock_universe,
    performance_cumprod,
)


def kelly_selection(
    adj_close_data: pd.DataFrame, risk_free_rate_u: float, n_days: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Calculate the leverage factor for the stock universe
    leverage = kelly_stock_universe(
        adj_close_data.iloc[-(n_days + 1) :], risk_free_rate_u, n_days
    )
    # Apply filters to the leverage factor
    lev_greater_10 = leverage[leverage > 10]
    largest_20 = leverage.nlargest(20).round(2)
    largest_20.name = "Kelly"

    return leverage, lev_greater_10, largest_20


def adr_selection(
    high_data: pd.DataFrame, low_data: pd.DataFrame, n_days: int
) -> tuple[pd.Series, pd.Series]:
    # Calculate the average daily range
    daily_range = (high_data.iloc[-n_days:] / low_data.iloc[-n_days:]).mean()
    average_daily_range = (100 * (daily_range - 1)).round(2)
    average_daily_range.name = "ADR [%]"

    return daily_range, average_daily_range


def volatility_selection(adj_close_data: pd.DataFrame, n_days: int) -> pd.Series:
    # Compute the volatility for the stock universe
    volatility = empirical_annualized_volatility(adj_close_data, window=n_days)

    return volatility.iloc[0].round(2)


def positive_return_selection(
    adj_close_data: pd.DataFrame, n_days: int
) -> tuple[pd.Series, pd.Index]:
    # Compute the returns for the stock universe
    returns = performance_cumprod(
        adj_close_data.iloc[-(n_days + 1) :].pct_change()
    ).round(2)
    returns.name = "Return [%]"
    # Filter for the largest ADR in the past n_days with positive trend
    returns_greater_10 = returns.index[returns > 10]

    return returns, returns_greater_10
