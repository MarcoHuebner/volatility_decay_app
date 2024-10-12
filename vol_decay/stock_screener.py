"""
Defines preprocessing functions for stock selection based on the Kelly criterion, 
average daily range, volatility, and positive returns.

"""

import pandas as pd

from src.utils.utils import performance_cumprod
from src.utils.volatility_calculations import empirical_annualized_volatility
from src.utils.kelly_calculations import kelly_stock_universe


def kelly_selection(
    adj_close_data: pd.DataFrame, risk_free_rate_u: float, n_days: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    # calculate the leverage factor for the stock universe
    leverage = kelly_stock_universe(
        adj_close_data.iloc[-(n_days + 1) :], risk_free_rate_u, n_days
    )
    if leverage.isnull().values.any() or leverage.empty:
        raise ValueError("The computed leverages contain missing values or are empty.")
    # apply filters to the leverage factor
    lev_greater_10 = leverage[leverage > 10]
    largest_20 = leverage.nlargest(20).round(2)
    largest_20.name = "Kelly"

    return leverage, lev_greater_10, largest_20


def compute_adr(
    high_data: pd.DataFrame, low_data: pd.DataFrame, n_days: int
) -> tuple[pd.Series, pd.Series]:
    # calculate the average daily range
    daily_range = (high_data.iloc[-n_days:] / low_data.iloc[-n_days:]).mean()
    if daily_range.isnull().values.any() or daily_range.empty:
        raise ValueError("The daily range contains missing values or is empty.")
    average_daily_range = (100 * (daily_range - 1)).round(2)
    average_daily_range.name = "ADR [%]"

    return daily_range, average_daily_range


def volatility_selection(adj_close_data: pd.DataFrame, n_days: int) -> pd.Series:
    # Compute the volatility for the stock universe
    volatility = empirical_annualized_volatility(adj_close_data, window=n_days)

    return volatility.iloc[-1].round(2)


def positive_return_selection(
    adj_close_data: pd.DataFrame, n_days: int
) -> tuple[pd.Series, pd.Index]:
    # Compute the returns for the stock universe
    returns = performance_cumprod(
        adj_close_data.iloc[-(n_days + 1) :].pct_change().dropna()
    ).round(2)
    if returns.isnull().values.any() or returns.empty:
        raise ValueError("The returns contain missing values or are empty.")
    returns.name = "Return [%]"
    # Filter for the largest ADR in the past n_days with positive trend
    returns_greater_10 = returns.index[returns > 10]

    return returns, returns_greater_10
