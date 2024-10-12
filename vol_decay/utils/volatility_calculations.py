"""
Different methods to calculate the (annualized) volatility of a stock.

"""

import numpy as np
import pandas as pd

from arch import arch_model
from deprecated import deprecated
from tqdm import tqdm

from vol_decay import constants
from vol_decay.utils.utils import validate_inputs


def empirical_annualized_volatility(
    data: pd.Series | pd.DataFrame, window: int = constants.trading_days
) -> float | pd.DataFrame:
    """
    Calculate the annualized volatility for every day in the stock data.

    :param data: pd.Series, price data
    :param window: int, window size for the rolling standard deviation
    :return: float, annualized volatility in percent
    """
    validate_inputs(data)
    # calculate the volatility
    volatility = data.pct_change().rolling(window=window).std().dropna()

    # convert to annualized volatility in percent
    return volatility * np.sqrt(constants.trading_days) * 100


@deprecated("Unused function, consider removing.")
def estimated_annualized_volatility(data_high: pd.Series, data_low: pd.Series) -> float:
    """
    Calculate the annualized volatility via an estimate of daily volatility
    based on "The Extreme Value Method for Estimating the Variance of the Rate of Return".
    Source: https://www.jstor.org/stable/2352357

    :param data: pd.Series, price data
    :return: float, annualized volatility in percent
    """
    for data in [data_high, data_low]:
        validate_inputs(data)
    # calculate the daily volatility
    daily_volatility = np.log(data_high / data_low).mean()

    # convert to annualized volatility in percent
    return daily_volatility * np.sqrt(constants.trading_days) * 100


@deprecated("Unused function, consider removing.")
def garch_estimated_volatility(data: pd.Series) -> float:
    """
    Use a GARCH model to forecast the annualized volatility every day.
    See https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html

    :param data: pd.Series, price data
    :return: float, forecasted annualized volatility
    """
    validate_inputs(data)
    # calculate the daily returns
    daily_returns = data.pct_change().dropna() * 100
    # create a GARCH model
    model_type = "EGARCH"
    forecasts = {}
    # Reduce the number of forecasts to speed up the computation
    reduction_factor = 3
    # Do recursive forecasting for the last year
    for i in tqdm(
        range(constants.trading_days // reduction_factor),
        desc=f"{model_type} forecasting",
    ):
        end_date = daily_returns.index[-constants.trading_days + i * reduction_factor]
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
        forecasts[temp.index[0]] = temp.iloc[0].values[0] * np.sqrt(
            constants.trading_days
        )

    return pd.Series(forecasts)


@deprecated("Unused function, consider removing.")
def empirical_var(data: pd.Series, alpha: float) -> float:
    """
    Calculate the Value at Risk (VaR) of the stock data.

    :param data: pd.Series, stock data
    :param alpha: float, confidence level
    :return: float, VaR
    """
    validate_inputs(data)
    # calculate the daily returns
    daily_returns = data.pct_change().dropna()

    # calculate the VaR
    return np.quantile(daily_returns, alpha)
