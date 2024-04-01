"""
Utility functions for the app.

"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance
from arch import arch_model
from tqdm import tqdm


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


def gmean(x: float, time_window: int = 1, days_in_year: int = 365):
    return (x + 1) ** (time_window / days_in_year) - 1


def performance_cumprod(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Calculate the cumulative product of the returns.

    :param returns: pd.Series | pd.DataFrame, daily returns
    :return: float | pd.Series, return on the last day in percent
    """
    return ((1 + returns).cumprod().iloc[-1] - 1) * 100


def simplified_lev_factor(
    daily_returns: pd.Series,
    expense_ratio: float,
    rel_transact_costs: float,
    leverage: float = 1.0,
    percent: float = 100.0,
):
    """
    Calculate the daily returns of a factor with leverage using a simplified model.

    :param daily_returns: pd.Series, daily returns of the underlying asset
    :param expense_ratio: float, expense ratio of the factor (in percent)
    :param rel_transact_costs: float, cost ratio assumed for each buy and sell transaction (in percent)
    :param leverage: float | pd.Series, leverage of the factor
    :param percent: float, percentage factor used for expense ratio conversion
    :return: pd.Series, daily returns of the factor with leverage
    """
    daily_returns = daily_returns * leverage + gmean(-expense_ratio / percent)

    # simplify: Assume the costs consist of only volume depend costs, neglecting fixed costs
    daily_returns.iloc[0] -= rel_transact_costs / percent
    daily_returns.iloc[-1] -= rel_transact_costs / percent

    return daily_returns


def simplified_knockout(
    price: pd.Series,
    expense_ratio: float,
    rel_transact_costs: float,
    initial_leverage: float,
    percent: float = 100,
) -> pd.Series:
    """
    Calculate the daily returns of a knockout product using a simplified model.
    Working with closing prices, this supposes the knockout was bought at the
    closing course of the first day, making zero returns on the first day.

    :param price: pd.Series, price of the underlying asset
    :param expense_ratio: float, expense ratio of the knockout product (in percent)
    :param rel_transact_costs: float, cost ratio assumed for each buy and sell transaction (in percent)
    :param initial_leverage: float, initial leverage factor of the knockout product
    :param percent: float, percentage factor used for expense ratio conversion
    :return: pd.Series, daily returns of the knockout product
    """
    # compute knockout barrier, incl. expense ratio estimation
    ko_val = (
        price.iloc[0] * (1 - (1 / initial_leverage)) * (1 - expense_ratio / percent)
    )
    # compute daily returns
    pct_change = (price - ko_val).pct_change()

    # get first knockout event (if it exists)
    mask = price.le(ko_val)
    index = mask.idxmax()

    if mask[index]:
        # set all following returns to zero to stay at the knockout level
        pct_change.loc[index:] = 0
    else:
        pass

    # simplify: Assume the costs consist of only volume depend costs, neglecting fixed costs
    pct_change.iloc[0] -= rel_transact_costs / percent
    pct_change.iloc[-1] -= rel_transact_costs / percent

    return pct_change.fillna(0)


def kelly_leverage(
    daily_returns: pd.Series,
    yearly_risk_free: float = 2.0,
    time_window: int = 60,
    safety: bool = True,
) -> pd.Series:
    """
    Compute the Kelly leverage fraction for a given time window based on the
    past 60 day returns and 60 day volatility.

    :param daily_returns: pd.Series, daily returns of the underlying asset
    :param yearly_risk_free: float, yearly risk-free rate in percent
    :param time_window: int, time window for the rolling average
    :param safety: bool, whether to apply safety margins
    :return: pd.Series, Kelly leverage fraction
    """
    # get rolling average returns and volatility, excluding the current day
    rolling_returns_estimate = (
        daily_returns.shift(1).rolling(window=time_window).mean() * time_window
    )
    rolling_volatility_estimate = (
        daily_returns.shift(1).rolling(window=time_window).std() * time_window**0.5
    )
    # get the risk-free rate for the time window
    time_window_risk_free = gmean(yearly_risk_free / 100, time_window=time_window)

    # add safety margins (underestimation of returns, overestimation of volatility)
    if safety:
        rolling_returns_estimate = rolling_returns_estimate - 0.02
        rolling_volatility_estimate = rolling_volatility_estimate + 0.03

    # calculate the Kelly leverage fraction
    leverage = (
        rolling_returns_estimate - time_window_risk_free
    ) / rolling_volatility_estimate**2

    # replace nan values with 1
    return leverage.fillna(1.0)


def kelly_stock_universe(
    data: pd.DataFrame, risk_free_rate: float, n_days: int
) -> pd.DataFrame:
    # Compute the Kelly criterion for the stock universe (simplified)
    change = data.pct_change()
    mean = change.mean() * n_days
    std = change.std() * n_days**0.5

    # get the risk-free rate for the time window
    time_window_risk_free = gmean(risk_free_rate / 100, time_window=n_days)

    # calculate the Kelly leverage fraction
    return (mean - time_window_risk_free) / std**2


def estimated_annualized_volatility(data_high: pd.Series, data_low: pd.Series) -> float:
    """
    Calculate the annualized volatility via an estimate of daily volatility
    based on "The Extreme Value Method for Estimating the Variance of the Rate of Return".
    Source: https://www.jstor.org/stable/2352357

    :param data: pd.Series, price data
    :return: float, annualized volatility in percent
    """
    # Calculate the daily volatility
    daily_volatility = np.log(data_high / data_low)

    # Convert to annualized volatility in percent
    return daily_volatility * np.sqrt(252) * 100


def empirical_annualized_volatility(data: pd.Series, window: int = 252) -> float:
    """
    Calculate the annualized volatility for every day in the stock data.

    :param data: pd.Series, price data
    :param window: int, window size for the rolling standard deviation
    :return: float, annualized volatility in percent
    """
    # Calculate the volatility
    volatility = data.pct_change().rolling(window=window).std().dropna()

    # Convert to annualized volatility in percent
    return volatility * np.sqrt(252) * 100


def garch_estimated_volatility(data: pd.Series) -> float:
    """
    Use a GARCH model to forecast the annualized volatility every day.
    See https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html

    :param data: pd.Series, price data
    :return: float, forecasted annualized volatility
    """
    # Calculate the daily returns
    daily_returns = data.pct_change().dropna() * 100
    # Create a GARCH model
    model_type = "EGARCH"
    forecasts = {}
    # Reduce the number of forecasts to speed up the computation
    reduction_factor = 3
    # Do recursive forecasting for the last year
    for i in tqdm(range(252 // reduction_factor), desc=f"{model_type} forecasting"):
        end_date = daily_returns.index[-252 + i * reduction_factor]
        am = arch_model(
            daily_returns.loc[:end_date],
            mean="AR",
            vol=model_type,
            p=1,
            o=0,
            q=1,
            dist="Normal",
        )
        res = am.fit(disp="off")
        # Get the volatility forecast
        temp = res.forecast(reindex=False).variance
        # Store and annualize the forecast
        forecasts[temp.index[0]] = temp.iloc[0].values[0] * np.sqrt(252)

    return pd.Series(forecasts)


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


# define yfinance functions
@st.cache_data(ttl=3600, show_spinner=False)  # cache for 1 hour
def fetch_ticker_data(ticker: str) -> dict[str, None | str | pd.Series | pd.DataFrame]:
    """
    Fetch the stock data from Yahoo Finance API via yfinance and calculate the
    50-day and 200-day moving average (MA).

    :param ticker: str, ticker symbol of the stock
    :return: dict, contains name, stock closing price, ann. vol., 30-day vol.,
                   50-day MA, 200-day MA, earnings dates (if available)
    """
    # Fetch the stock price data from Yahoo Finance API via yfinance
    lazy_dict = yfinance.Ticker(ticker)

    # Pick the relevant data
    data = lazy_dict.history(period="2y", interval="1d", auto_adjust=True)["Close"]
    if data.empty:
        raise ValueError(f"Ticker {ticker} not found. Please check the ticker symbol.")
    else:
        # Slice the data to the one and two years, to reduce computation time
        starting_date_1y = data.index[-252]
        # Calculate the moving averages
        ma50 = data.rolling(window=50).mean().dropna()
        ma200 = data.rolling(window=200).mean().dropna()
        # Get the full name of the ticker symbol
        name = lazy_dict.info["longName"]
        # Compute the 30-day volatility (VIX for S&P)
        vix = empirical_annualized_volatility(data, window=30)
        # Compute the annualized volatility
        ann_vol = empirical_annualized_volatility(data)
        # Get the earnings dates, force them None if KeyError occurs
        try:
            earnings_dates = lazy_dict.get_earnings_dates()
        except KeyError:
            earnings_dates = None
        # Create a dictionary with the last year of (trading days) data (except for price)
        result_dict = {
            "name": name,
            "price": data,
            "ann_volatility": ann_vol.loc[starting_date_1y:],
            "30_d_volatility_vix": vix.loc[starting_date_1y:],
            # "garch_volatility": garch_estimated_volatility(data),
            "ma50": ma50.loc[starting_date_1y:],
            "ma200": ma200.loc[starting_date_1y:],
            "earnings": earnings_dates,
        }
        return result_dict


@st.cache_data(ttl=3600 * 23.75, show_spinner=False)  # cache for a day minus overlap
def download_universe() -> pd.DataFrame:
    # Opening and reading the symbols file (currently 1148 symbols, 172 unavailable)
    symbols_list = open("assets/symbols_list.txt", "r")
    symbols = symbols_list.read().split()
    symbols_list.close()

    # Download the data for all (available) symbols in the universe
    # Takes about 3 minutes for around 1000 symbols
    download = yfinance.download(symbols, period="3mo", interval="1d")

    return download.dropna(axis=1)


# define the helper functions
def xaxis_slider(
    reference_data: pd.Series,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """
    Create a slider for the x-axis of the plotly graph.

    :return: dict, slider for the x-axis
    """
    current_date = reference_data.index[-1]

    # define the slider
    xaxis_with_slider = dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
        range=[current_date - timedelta(days=365), current_date],
    )

    return xaxis_with_slider


def plot_earnings_dates(
    earnings: pd.Series, reference_data: pd.Series, fig: go.Figure
) -> go.Figure:
    """
    Add vertical dashed lines for each earning date within the past
    365 and upcoming 90 days to the figure.

    :param earnings: pd.Series, the earnings dates
    :param reference_data: pd.Series, the reference data
    :param fig: go.Figure, the plotly figure
    :return: go.Figure, the updated plotly figure
    """
    # Filter earnings dates to the last year and the next 90 days
    dates = earnings.index
    current_date = reference_data.index[-1]
    start_date = current_date - timedelta(days=365)
    end_date = current_date + timedelta(days=90)
    filtered_dates = dates[(dates >= start_date) & (dates <= end_date)]

    # Add dummy scatter trace for the legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="LightSeaGreen", width=2, dash="dash"),
            name="Earnings",
        )
    )

    # Plot vertical dashed lines for each date
    for date in filtered_dates:
        fig.add_shape(
            type="line",
            x0=date,
            x1=date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="LightSeaGreen",
                width=2,
                dash="dash",
            ),
        )

    return fig
