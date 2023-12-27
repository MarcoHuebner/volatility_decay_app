"""
Utility functions for the app.

"""

import numpy as np
import pandas as pd
import yfinance

from typing import Union


# define the mathematical functions
def leveraged_return(
    lev_factor: float,
    cagr_underlying: float,
    leverage_expense_ratio: float,
    libor: float,
    yearly_volatility: float,
) -> float:
    """
    Calculate the leveraged return according to
    https://www.reddit.com/r/HFEA/comments/tue7n6/the_volatility_decay_equation_with_verification/

    :param lev_factor: float, leverage factor applied
    :param cagr_underlying: float, compound annual growth rate
    :param leverage_expense_ratio: float, expense ratio of the leveraged position (fund)
    :param libor: float, average LIBOR during investment period + 0.4%
    :param yearly_volatility: float, annualized volatility
    :return: float, annual return of leveraged position (fund)
    """
    # short names/ notation
    x = lev_factor
    r = cagr_underlying
    E = leverage_expense_ratio
    I = libor
    s = yearly_volatility / np.sqrt(252)  # get daily volatility

    # define helpful quantities to avoid repitition & better overview
    exp = np.exp(np.log(1 + r) / 252)
    e_i = (E + 1.1 * (x - 1) * I) / 252
    first = x * s + x * s**2 / (2 * exp) + x * exp - e_i - x + 1
    second = x * exp**2 / (s + 0.5 * s**2 * exp ** (-1) + exp) - e_i - x + 1

    return (first * second) ** 126 - 1


def leveraged_return_mesh(
    lev: float, cagr_undr: float, exp: float, lib: float, vol_undr: float
) -> np.ndarray:
    """
    Create a mesh of leveraged returns for visualizing the leveraged return against
    the underlying return and against the volatility. Return shows quadratic behaviour
    similar to the some discussion in the following sources:
    - https://www.afrugaldoctor.com/home/leveraged-etfs-and-volatility-decay-part-2
    - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1664823

    :param lev: float, leverage factor applied
    :param cagr_undr: float, compound annual growth rate in percent
    :param exp: float, expense ratio of the leveraged position (fund) in percent
    :param lib: float, average LIBOR during investment period in percent
    :param vol_undr: float, annualized volatility of the underlying in percent
    :return: np.ndarray, volatility & underlying CAGR leveraged return mesh
    """
    # create mesh leveraged CAGR as array
    mesh = np.zeros((len(vol_undr), len(cagr_undr)))

    for i, vol in enumerate(vol_undr):
        for j, cagr in enumerate(cagr_undr):
            # reflect on volatility axis due to the way, plotly sets-up heatmaps
            # also, rescale percentage values, as otherwise not readable in the sliders
            mesh[i, j] = (
                leveraged_return(lev, cagr / 100, exp / 100, lib / 100, vol / 100)
                - cagr / 100
            )

    return np.round(mesh * 100, 2)


# define yfinance functions
def fetch_ticker_data(ticker: str) -> dict[str, Union[str, pd.Series]]:
    """
    Fetch the stock data from Yahoo Finance API via yfinance and calculate the
    50-day and 200-day moving average (MA).

    :param ticker: str, ticker symbol of the stock
    :return: dict[str, Union[str,pd.Series]], name, stock closing, daily high and low
                                              price data, 50-day MA, 200-day MA
    """
    # Fetch the stock price data from Yahoo Finance API via yfinance
    lazy_dict = yfinance.Ticker(ticker)

    # Pick the relevant data
    data = lazy_dict.history(period="1y", interval="1d", auto_adjust=True)["Close"]
    data_high = lazy_dict.history(period="1y", interval="1d", auto_adjust=True)["High"]
    data_low = lazy_dict.history(period="1y", interval="1d", auto_adjust=True)["Low"]
    if data.empty:
        raise ValueError(f"Ticker {ticker} not found. Please check the ticker symbol.")
    else:
        # Calculate the moving averages
        ma50 = data.rolling(window=50).mean().dropna()
        ma200 = data.rolling(window=200).mean().dropna()
        # Get the full name of the ticker symbol
        name = lazy_dict.info["longName"]
        # Create a dictionary with the relevant data
        result_dict = {
            "name": name,
            "data": data,
            "data_high": data_high,
            "data_low": data_low,
            "ma50": ma50,
            "ma200": ma200,
        }
        return result_dict


def empirical_annualized_volatility(data_high: pd.Series, data_low: pd.Series) -> float:
    """
    Calculate the annualized volatility via an estimate of daily volatility
    based on V_p in "Modelling Volatility Using High, Low, Open and Closing
    Prices: Evidence from Four S&P Indices".
    Source: https://core.ac.uk/download/pdf/52391988.pdf

    :param data: pd.Series, stock data
    :return: float, annualized volatility
    """
    # Calculate the daily volatility
    daily_volatility = 0.361 * (np.log(data_high / data_low)) ** 2
    # Convert to annualized volatility
    return daily_volatility * np.sqrt(252)


def empirical_var(data: pd.Series, alpha: float) -> float:
    """
    Calculate the Value at Risk (VaR) of the stock data.

    :param data: pd.Series, stock data
    :param alpha: float, confidence level
    :return: float, VaR
    """
    # Calculate the daily returns
    daily_returns = data.pct_change().dropna()

    # Calculate the VaR
    return np.quantile(daily_returns, alpha)
