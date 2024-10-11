"""
Define functions to fetch and cache stock data from Yahoo Finance API via yfinance,
and calculate forecasts using Prophet.

"""

import pandas as pd
import streamlit as st
import yfinance

from prophet import Prophet

from src import constants
from src.utils.volatility_calculations import empirical_annualized_volatility


def fit_prophet_model(price: pd.Series, periods: int) -> pd.DataFrame:
    # define and fit the Prophet model to the price time series data
    model = Prophet(interval_width=0.95)
    model.fit(pd.DataFrame({"ds": price.index.tz_localize(None), "y": price.values}))
    # create a future dataframe and forecast the given period
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    # add the timezone back to the "ds" column
    forecast["ds"] = forecast["ds"].dt.tz_localize("Etc/GMT+4")

    return forecast


@st.cache_data(ttl=1800, show_spinner=False)
def get_prophet_forecast(price: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit two Prophet models to the price data for different lookback windows
    and forecast the next days of price development.

    :param price: The price data to fit the Prophet models to.
    :return: tuple, two forecast dataframes
    """
    current_forecast = fit_prophet_model(price, 10)
    previous_forecast = fit_prophet_model(price[:-5], 15)

    return current_forecast, previous_forecast


# define yfinance functions
def fetch_earnings_dates(lazy_dict: yfinance.Ticker) -> pd.DataFrame | None:
    try:
        return lazy_dict.get_earnings_dates()
    except KeyError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker: str) -> dict[str, None | str | pd.Series | pd.DataFrame]:
    """
    Fetch the stock data from Yahoo Finance API via yfinance, cache it for one hour
    and calculate the 50-day and 200-day moving average (MA).

    :param ticker: str, ticker symbol of the stock
    :return: dict, contains name, stock closing price, ann. vol., 30-day vol.,
                   50-day MA, 200-day MA, earnings dates (if available)
    """
    # Fetch the stock price data from Yahoo Finance API via yfinance
    lazy_dict = yfinance.Ticker(ticker)

    # Pick the relevant data
    data = lazy_dict.history(period="6y", interval="1d", auto_adjust=True)
    if data.empty:
        raise ValueError(f"Ticker {ticker} not found. Please check the ticker symbol.")

    # Collect closing prices and daily low prices (forward-fill, drop other missing values)
    closing = data["Close"].ffill().dropna()
    low = data["Low"].ffill().dropna()
    # Align the closing and low prices (drop the days with missing values in either column)
    # low, closing = low.align(closing, join="inner")
    # Slice the data to the one and two years, to reduce computation time
    starting_date_1y = closing.index[-constants.five_years]
    # Calculate the moving averages
    ma50 = closing.rolling(window=50).mean().dropna()
    ma200 = closing.rolling(window=200).mean().dropna()
    # Get the full name of the ticker symbol
    name = lazy_dict.info["longName"]
    # Compute the 30-day volatility (VIX for S&P)
    vix = empirical_annualized_volatility(closing, window=30)
    # Compute the annualized volatility
    ann_vol = empirical_annualized_volatility(closing)
    # Get the earnings dates, force them None if KeyError occurs
    earnings_dates = fetch_earnings_dates(lazy_dict)
    # Create a dictionary with the last year of (trading days) data (except for price)
    result_dict = {
        "name": name,
        "price": closing,
        "low": low,
        "ann_volatility": ann_vol.loc[starting_date_1y:],
        "30_d_volatility_vix": vix.loc[starting_date_1y:],
        # "garch_volatility": garch_estimated_volatility(data),
        "ma50": ma50.loc[starting_date_1y:],
        "ma200": ma200.loc[starting_date_1y:],
        "earnings": earnings_dates,
    }
    return result_dict


@st.cache_data(ttl=3600 * 23.75, show_spinner=False)
def download_universe() -> pd.DataFrame:
    """
    Download the data for all (available) symbols in the universe and cache data for
    one day minus 15 minutes overlap.

    :return: pd.DataFrame, stock data for all symbols in the universe
    """
    # open and read the symbols file (currently 1148 symbols, 172 unavailable)
    symbols_list = open("assets/symbols_list.txt", "r")
    symbols = symbols_list.read().split()
    symbols_list.close()

    # download the data for all (available) symbols in the universe
    # takes about 3 minutes for around 1000 symbols
    download = yfinance.download(symbols, period="3mo", interval="1d")

    # fill missing values with preceding values (at most three) and then with subsequent
    # values (at most three) e.g. to cover for country-specific public holidays
    download = download.ffill(axis=1, limit=3).bfill(axis=1, limit=3)
    # remove remaining columns with NaNs
    return download.dropna(axis=1)
