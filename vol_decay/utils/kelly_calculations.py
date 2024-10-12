"""
Kelly criterion calculations for different levels of complexity data.

"""

import numpy as np
import pandas as pd

from vol_decay.utils.utils import gmean, validate_inputs

pct = 100.0


def kelly_crit(
    yearly_er: float | pd.Series | pd.DataFrame,
    yearly_risk_free: float | pd.Series | pd.DataFrame,
    yearly_volatility: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the Kelly Criterion for float, pd.Series, or pd.DataFrame inputs.

    :param yearly_er: float, pd.Series, or pd.DataFrame, yearly expected return in percent
    :param yearly_risk_free: float, pd.Series, or pd.DataFrame, yearly risk-free rate in percent
    :param yearly_volatility: float, pd.Series, or pd.DataFrame, yearly volatility in percent
    :return: float, pd.Series, or pd.DataFrame, Kelly Criterion
    """
    if isinstance(yearly_volatility, (pd.Series, pd.DataFrame)):
        if not (yearly_volatility.dropna() > 0).all().all():
            raise ValueError("Volatility must be positive.")
    elif isinstance(yearly_volatility, (float, int)):
        if not yearly_volatility > 0:
            raise ValueError("Volatility must be positive.")
    else:
        raise TypeError(
            f"yearly_volatility must be a float, pd.Series, or pd.DataFrame but found {type(yearly_volatility)}."
        )
    
    # Ensure all inputs have the same length
    if any(isinstance(data, (pd.Series, pd.DataFrame)) for data in (yearly_er, yearly_risk_free)):
        lengths = [len(data) for data in (yearly_er, yearly_risk_free, yearly_volatility) if isinstance(data, (pd.Series, pd.DataFrame))]
        if len(set(lengths)) != 1:
            raise ValueError("All inputs must have the same length")
        
    # NOTE: the factor of 100 corrects for the percentage values
    return pct * (yearly_er - yearly_risk_free) / yearly_volatility**2


def kelly_crit_mesh(
    yearly_er: float, yearly_risk_free: float, yearly_volatility: float
) -> np.ndarray:
    """
    Calculate the Kelly Criterion meshed for different underlying CAGR and volatility.

    :param yearly_er: float, yearly expected return in percent
    :param yearly_risk_free: float, yearly risk-free rate in percent
    :param yearly_volatility: float, yearly volatility in percent
    :return: np.ndarray, Kelly Criterion mesh
    """
    mesh = np.zeros((len(yearly_volatility), len(yearly_er)))
    for i, vol in enumerate(yearly_volatility):
        for j, cagr in enumerate(yearly_er):
            # reflect on volatility axis due to the way, plotly sets-up heatmaps
            # also, rescale percentage values, as otherwise not readable in the sliders
            mesh[i, j] = kelly_crit(cagr, yearly_risk_free, vol)
    return np.round(mesh, 2)


def kelly_leverage(
    daily_returns: pd.Series,
    yearly_risk_free: float = 2.0,
    time_window: int = 60,
    safety_margin: bool = True,
) -> pd.Series:
    """
    Compute the Kelly leverage fraction for a given time window based on the
    past 60 day returns and 60 day volatility.

    :param daily_returns: pd.Series, daily returns of the underlying asset
    :param yearly_risk_free: float, yearly risk-free rate in percent
    :param time_window: int, time window for the rolling average
    :param safety_margin: bool, whether to apply safety margins to the returns and volatility
    :return: pd.Series, Kelly leverage fraction
    """
    validate_inputs(daily_returns)
    # get rolling average returns and volatility, excluding the current day
    rolling_returns_estimate = (
        daily_returns.shift(1).rolling(window=time_window).mean() * time_window
    )
    rolling_volatility_estimate = (
        daily_returns.shift(1).rolling(window=time_window).std() * time_window**0.5
    )
    # get the risk-free rate for the time window
    time_window_risk_free = gmean(yearly_risk_free / pct, time_window=time_window)

    # add safety margins (underestimation of returns, overestimation of volatility)
    if safety_margin:
        rolling_returns_estimate = rolling_returns_estimate - 0.02
        rolling_volatility_estimate = rolling_volatility_estimate + 0.03
        rolling_volatility_estimate = rolling_volatility_estimate.clip(lower=0.0001)

    # calculate the Kelly leverage fraction
    leverage = (
        kelly_crit(
            rolling_returns_estimate,
            time_window_risk_free,
            rolling_volatility_estimate,
        )
        / pct
    )

    # replace NaN values with 1
    return leverage.fillna(1.0)


def kelly_stock_universe(
    data: pd.DataFrame, yearly_risk_free: float, n_days: int
) -> pd.DataFrame:
    """
    Compute the Kelly Criterion for a stock universe.

    :param data: pd.DataFrame, stock price data
    :param yearly_risk_free: float, yearly risk-free rate in percent
    :param n_days: int, number of days for the time window
    :return: pd.DataFrame, Kelly leverage fraction for the stock universe
    """
    # remove columns with NaN values
    data = data.dropna(axis=1)
    validate_inputs(data)
    # compute the Kelly criterion for the stock universe (simplified)
    change = data.pct_change().dropna()
    mean = change.mean() * n_days
    std = change.std() * n_days**0.5

    # get the risk-free rate for the time window
    time_window_risk_free = gmean(yearly_risk_free / pct, time_window=n_days)

    # calculate the Kelly leverage fraction
    return kelly_crit(mean, time_window_risk_free, std) / pct
