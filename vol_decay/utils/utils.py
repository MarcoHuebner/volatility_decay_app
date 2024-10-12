"""
Utility functions for the app.

"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vol_decay import constants


def validate_inputs(series: pd.Series | pd.DataFrame) -> None:
    if series.isnull().values.any() or series.empty:
        raise ValueError("The input data contains missing values or is empty.")


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
    s = yearly_volatility / np.sqrt(constants.trading_days)  # get daily volatility

    # define helpful quantities to avoid repitition & better overview
    exp = np.exp(np.log(1 + r) / constants.trading_days)
    e_i = (E + 1.1 * (x - 1) * I) / constants.trading_days
    first = x * s + x * s**2 / (2 * exp) + x * exp - e_i - x + 1
    second = x * exp**2 / (s + 0.5 * s**2 * exp ** (-1) + exp) - e_i - x + 1

    return (first * second) ** (constants.trading_days // 2) - 1


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
    validate_inputs(returns)

    # seemingly no (time) performance gains with np.cumprod here
    return ((1 + returns).cumprod().iloc[-1] - 1) * 100


# define plotting helper functions
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
    validate_inputs(earnings.index)
    # filter earnings dates to the last year and the next 90 days
    dates = earnings.index
    current_date = reference_data.index[-1]
    start_date = current_date - timedelta(days=365)
    end_date = current_date + timedelta(days=90)
    filtered_dates = dates[(dates >= start_date) & (dates <= end_date)]

    # add dummy scatter trace for the legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="LightSeaGreen", width=2, dash="dash"),
            name="Earnings",
        )
    )

    # plot vertical dashed lines for each date
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


# TODO: Add automatic download of the universe every day (best: At night)
"""def repeat_download_universe():
    while True:
        download_universe()
        # Sleep for 24 hours +/- up to 15 minutes
        time.sleep(24 * 3600 + random.randint(-15 * 60, 15 * 60))

# Start the background thread
threading.Thread(target=repeat_download_universe, daemon=True).start()

download_universe()"""
