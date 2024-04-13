"""
Define the functions to update the plots for the investments page.

"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import constants
from src.utils import (
    fetch_ticker_data,
    get_prophet_forecast,
    kelly_crit,
    kelly_leverage,
    performance_cumprod,
    plot_earnings_dates,
    simplified_knockout,
    simplified_lev_factor,
    xaxis_slider,
)

# define colours, loosely related to the streamlit default colours
# https://discuss.streamlit.io/t/access-streamlit-default-color-palette/35737
st_blue = "#83c9ff"
st_dark_blue = "#0068c9"
st_darker_blue = "#0054a3"
st_red = "#ff2b2b"
st_green = "#21c354"


def forecast_band_plot(
    fig: go.Figure, forecast: pd.DataFrame, color: str, name: str
) -> go.Figure:
    # add lower bound of forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            name="CI Lower Bound",
            line=dict(color=color),
            opacity=0.5,
            visible="legendonly",
            showlegend=False,
            legendgroup=name,
        )
    )
    # add upper bound of forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            name="CI Upper Bound",
            line=dict(color=color),
            fill="tonexty",
            opacity=0.5,
            visible="legendonly",
            showlegend=False,
            legendgroup=name,
        )
    )
    # add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name=name,
            visible="legendonly",
            line=dict(color=color),
            legendgroup=name,
        )
    )

    return fig


# define the ticker price data plot
def update_ticker_plot(ticker: str, risk_free_rate_ticker: float) -> go.Figure:
    # create the plotly figure
    fig = go.Figure()
    # define colours, loosely related to the streamlit default colours
    # https://discuss.streamlit.io/t/access-streamlit-default-color-palette/35737
    st_blue = "#83c9ff"
    st_dark_blue = "#0068c9"
    st_darker_blue = "#0054a3"
    st_red = "#ff2b2b"
    st_light_red = "#ff8c8c"
    st_green = "#21c354"

    # get data
    result_dict = fetch_ticker_data(ticker)
    price = result_dict["price"].tail(constants.trading_days)
    # get earning dates if not None (e.g. for indices)
    if result_dict["earnings"] is not None:
        earnings = result_dict["earnings"]["Reported EPS"]
    # get prophet forecast
    forecast_current, forecast_old = get_prophet_forecast(price)

    # add price line
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name="Closing Price",
            line=dict(color=st_blue),
        )
    )
    # add forecast with uncertainty bands
    fig = forecast_band_plot(fig, forecast_current, st_darker_blue, "Current Forecast")
    fig = forecast_band_plot(fig, forecast_old, st_dark_blue, "Previous Forecast")
    # add 50-day moving average
    fig.add_trace(
        go.Scatter(
            x=result_dict["ma50"].index,
            y=result_dict["ma50"],
            name="50 Day MA",
            line=dict(color=st_dark_blue, dash="dash"),
        )
    )
    # add 200-day moving average
    fig.add_trace(
        go.Scatter(
            x=result_dict["ma200"].index,
            y=result_dict["ma200"],
            name="200 Day MA",
            line=dict(color=st_darker_blue, dash="dot"),
        )
    )
    # add annualized volatility
    fig.add_trace(
        go.Scatter(
            x=result_dict["ann_volatility"].index,
            y=result_dict["ann_volatility"],
            mode="lines",
            name="Annualized Volatility",
            yaxis="y2",
            line=dict(color=st_red),
        )
    )
    # add 30-d volatility (VIX for S&P)
    fig.add_trace(
        go.Scatter(
            x=result_dict["30_d_volatility_vix"].index,
            y=result_dict["30_d_volatility_vix"],
            mode="lines",
            name="30 Day Volatility Estimate",
            yaxis="y2",
            visible="legendonly",
            line=dict(color=st_red, dash="dot"),
        )
    )
    # add garch volatility
    fig.add_trace(
        go.Scatter(
            x=[result_dict["ann_volatility"].index[0]],
            y=[result_dict["ann_volatility"].min()],
            mode="text",
            name="GARCH Forecast Volatility",
            text=["Discontinued, see source code."],
            textfont=dict(size=10, color=st_light_red),
            textposition="top right",
            yaxis="y2",
            visible="legendonly",
        )
    )
    # add daily percentage change
    pct_change = result_dict["price"].pct_change().dropna() * 100
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=pct_change.iloc[-constants.trading_days :],
            mode="lines",
            name="Daily Returns",
            yaxis="y3",
            visible="legendonly",
            line=dict(color=st_green),
        )
    )
    if result_dict["earnings"] is not None:
        # add earnings dates
        fig = plot_earnings_dates(earnings, price, fig)

    # calculate the Kelly Criterion with maximum of the three volatilities
    average_vol_30d = result_dict["30_d_volatility_vix"].iloc[-52:].mean()
    average_daily_return = pct_change.iloc[-constants.trading_days :].mean()
    max_vol = max(
        result_dict["ann_volatility"].iloc[-1],
        # result_dict["garch_volatility"].iloc[-1],
        average_vol_30d,
    )
    # add a safety margin of -2% ann. return and +3% ann. volatility
    safety_return = -2
    safety_vol = 3
    kelly = kelly_crit(
        average_daily_return * constants.trading_days + safety_return,
        risk_free_rate_ticker,
        max_vol + safety_vol,
    )
    # calculate the leverage factor for 20% volatility
    lev_20 = 20 / result_dict["ann_volatility"].iloc[-1]
    # calculate the Percentage at Risk (PaR)
    par_5 = np.percentile(pct_change.dropna(), 5)
    par_1 = np.percentile(pct_change.dropna(), 1)

    # update layout
    fig.update_layout(
        title=f"<span style='font-size: 24px;'>Current Price and"
        + f" Volatility of {result_dict['name']}</span><br>"
        + f"<span style='font-size: 16px;'>Kelly Leverage Factor: {kelly:.2f}"
        + f" - Leverage @20% Volatility: {lev_20:.2f}"
        + f" - Current 1%/5% PaR (past 2y): {par_1:.2f}%/{par_5:.2f}%</span>",
        hovermode="x unified",
        yaxis=dict(
            title="Closing Prices", title_font=dict(color=st_blue), hoverformat=".2f"
        ),
        yaxis2=dict(
            title="Annualized Volatility [%]",
            side="right",
            overlaying="y",
            title_font=dict(color=st_red),
            hoverformat=".2f",
        ),
        yaxis3=dict(
            title="Daily Returns [%]",
            side="right",
            overlaying="y",
            position=0.96,
            title_font=dict(color=st_green),
            hoverformat=".2f",
        ),
        xaxis=xaxis_slider(price),
    )
    # update x width
    fig.update_xaxes(
        domain=[0.0, 0.89],
    )

    return fig


def get_derivatives_data(
    ticker: str,
    risk_free_rate_ticker: float,
    leverage: float,
    expenses: float,
    rel_transact_costs: float,
    time_window: int,
    holding_period: int,
    include_tax: bool,
) -> dict[str, pd.Series | list | float]:
    # get data of the underlying
    result_dict = fetch_ticker_data(ticker)
    price = result_dict["price"].tail(constants.trading_days)
    low = result_dict["low"].tail(constants.trading_days)
    pct_change = result_dict["price"].pct_change().dropna()
    pct_change_low = (low - price).pct_change().dropna()
    # get earning dates if not None (e.g. for indices)
    if result_dict["earnings"] is not None:
        earnings = result_dict["earnings"]["Reported EPS"]
    else:
        earnings = None

    # define the data dictionary
    data_dict = {"name": result_dict["name"], "price": price, "earnings": earnings}

    # how have derivatives with kelly criterion > 5 performed in the past?
    # show results of fixed length holding_period day intervals
    kelly_lev = kelly_leverage(
        pct_change, risk_free_rate_ticker, time_window=time_window
    ).tail(constants.trading_days)
    pct_change = pct_change.tail(constants.trading_days)
    pct_change_low = pct_change_low.tail(constants.trading_days)
    # get days on which the kelly criterion was > 5
    dates_iloc = np.where(kelly_lev > 5)[0]
    # add leverage and dates to the return dictionary
    data_dict["kelly_lev"] = kelly_lev
    data_dict["dates_iloc"] = dates_iloc
    if dates_iloc.size == 0:
        # set to 0 if no signals
        returns_1x, returns_lev, returns_ko = [0], [0], [0]
        win_ratio_ko, win_ratio_f = 0, 0
        reward_ko, reward_f = 0, 0
    else:
        # get all possible holding_period day interval returns
        returns_1x = [
            performance_cumprod(pct_change.iloc[date : date + holding_period])
            for date in dates_iloc
        ]
        returns_lev = [
            performance_cumprod(
                simplified_lev_factor(
                    pct_change.iloc[date : date + holding_period],
                    pct_change_low.iloc[date : date + holding_period],
                    expenses,
                    rel_transact_costs,
                    holding_period,
                    leverage,
                )
            )
            for date in dates_iloc
        ]
        returns_ko = [
            performance_cumprod(
                # assume that the knockout is bought during the day for
                # the closing price of the previous day
                simplified_knockout(
                    price.iloc[date - 1 : date + holding_period],
                    low.iloc[date - 1 : date + holding_period],
                    expenses,
                    rel_transact_costs,
                    holding_period,
                    leverage,
                )
            )
            for date in dates_iloc
        ]

        if include_tax:
            # define tax rate
            tax = 0.25
            solidary_tax = 0.01375
            remaining_gain = 1 - (tax + solidary_tax)
            # add tax on profits
            returns_lev = [r if r < 0 else r * remaining_gain for r in returns_lev]
            returns_ko = [r if r < 0 else r * remaining_gain for r in returns_ko]
            returns_lev = [r if r < 0 else r * remaining_gain for r in returns_lev]

        # pre-compute quantities to simplify the code and catch division by zero
        def avg_return(returns: list[float]) -> float:
            if any(math.isnan(x) for x in returns):
                raise ValueError("returns contains NaN values")
            return sum(returns) / max(len(returns), 1)

        pos_returns_ko = [r for r in returns_ko if r > 0]
        avg_win_ko = avg_return(pos_returns_ko)
        neg_returns_ko = [r for r in returns_ko if r <= 0]
        avg_loss_ko = avg_return(neg_returns_ko)
        pos_returns_f = [r for r in returns_lev if r > 0]
        avg_win_f = avg_return(pos_returns_f)
        neg_returns_f = [r for r in returns_lev if r <= 0]
        avg_loss_f = avg_return(neg_returns_f)
        # calculate win ratios and reward ratios
        win_ratio_ko = len(pos_returns_ko) / max(len(returns_ko), 1) * 100
        win_ratio_f = len(pos_returns_f) / max(len(returns_lev), 1) * 100
        reward_ko = avg_win_ko / max(abs(avg_loss_ko), 0.001)
        reward_f = avg_win_f / max(abs(avg_loss_f), 0.001)

    # add returns to the return dictionary
    data_dict["returns_1x"] = returns_1x
    data_dict["returns_lev"] = returns_lev
    data_dict["returns_ko"] = returns_ko
    # add win ratios and reward ratios to the return dictionary
    data_dict["win_ratio_ko"] = win_ratio_ko
    data_dict["win_ratio_f"] = win_ratio_f
    data_dict["reward_ko"] = reward_ko
    data_dict["reward_f"] = reward_f

    # calculate opacities based on comparison of returns
    opacities_lev = [
        0.3 if lev <= ko else 1.0 for lev, ko in zip(returns_lev, returns_ko)
    ]
    opacities_ko = [
        0.3 if ko < lev else 1.0 for lev, ko in zip(returns_lev, returns_ko)
    ]

    # add opacities to the return dictionary
    data_dict["opacities_lev"] = opacities_lev
    data_dict["opacities_ko"] = opacities_ko

    return data_dict


# define the past leverage and knock out returns plot
def update_derivatives_performance_plot(
    data_dict: dict[str, pd.Series | list | float],
    leverage: float,
    holding_period: int,
) -> go.Figure:
    # create the plotly figure
    fig = go.Figure()

    # unpack the data dictionary
    name = data_dict["name"]
    price = data_dict["price"]
    earnings = data_dict["earnings"]
    kelly_lev = data_dict["kelly_lev"]
    dates_iloc = data_dict["dates_iloc"]
    returns_1x = data_dict["returns_1x"]
    returns_lev = data_dict["returns_lev"]
    returns_ko = data_dict["returns_ko"]
    opacities_lev = data_dict["opacities_lev"]
    opacities_ko = data_dict["opacities_ko"]

    # get prophet forecast
    forecast_current, forecast_old = get_prophet_forecast(price)

    # add price line
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            mode="lines",
            name="Closing Price",
            line=dict(color=st_blue),
        )
    )
    # add forecast with uncertainty bands
    fig = forecast_band_plot(fig, forecast_current, st_darker_blue, "Current Forecast")
    fig = forecast_band_plot(fig, forecast_old, st_dark_blue, "Previous Forecast")
    # add unleveraged returns
    fig.add_trace(
        go.Scatter(
            x=price.index[dates_iloc],
            y=returns_1x,
            mode="markers",
            name="Underlying",
            yaxis="y2",
            marker=dict(color=st_green, symbol="circle"),
        )
    )
    # add leveraged factor returns
    fig.add_trace(
        go.Scatter(
            x=price.index[dates_iloc],
            y=returns_lev,
            mode="markers",
            name=f"{leverage}x Factor",
            yaxis="y2",
            marker=dict(
                color=st_dark_blue, symbol="triangle-up", opacity=opacities_lev
            ),
        )
    )
    # add knockout returns
    fig.add_trace(
        go.Scatter(
            x=price.index[dates_iloc],
            y=returns_ko,
            mode="markers",
            name=f"{leverage}x Knockout",
            yaxis="y2",
            marker=dict(color=st_darker_blue, symbol="square", opacity=opacities_ko),
        )
    )
    # add zero line
    fig.add_shape(
        type="line",
        xref="x",
        yref="y2",
        x0=price.index[0],
        y0=0,
        x1=price.index[-1],
        y1=0,
        line=dict(
            color=st_darker_blue,
            width=2,
            dash="dash",
        ),
    )
    # add past holding_period day cut-off line
    fig.add_shape(
        type="line",
        xref="x",
        yref="y2",
        x0=price.index[-holding_period],
        y0=min(min(returns_1x), min(returns_lev), min(returns_ko)),
        x1=price.index[-holding_period],
        y1=max(max(returns_1x), max(returns_lev), max(returns_ko)),
        line=dict(
            color=st_darker_blue,
            width=2,
            dash="dash",
        ),
    )
    # add leverage factor
    fig.add_trace(
        go.Scatter(
            x=kelly_lev.index,
            y=kelly_lev,
            mode="lines",
            name="Kelly Leverage Factor",
            yaxis="y3",
            line=dict(color=st_red),
            visible="legendonly",
            legendgroup="group1",
        )
    )
    # add typical leverage cutoff
    fig.add_trace(
        go.Scatter(
            x=[kelly_lev.index.min(), kelly_lev.index.max()],
            y=[5, 5],
            mode="lines",
            yaxis="y3",
            line=dict(color=st_red, dash="dash"),
            visible="legendonly",
            legendgroup="group1",
            showlegend=False,
        )
    )
    if earnings is not None:
        # add earnings dates
        fig = plot_earnings_dates(earnings, price, fig)

    # update layout
    fig.update_layout(
        title=f"<span style='font-size: 24px;'>{name} Derivative Performance</span><br>"
        + "<span style='font-size: 16px;'>Given a Rolling Kelly Leverage Factor of 5"
        + f" as Signal - Amount of Signals in the Past {constants.trading_days}"
        + f" Trading Days: {len(dates_iloc)}</span>",
        hovermode="x unified",
        yaxis=dict(
            title="Closing Prices", title_font=dict(color=st_blue), hoverformat=".2f"
        ),
        yaxis2=dict(
            title=f"Returns @ Buy Date + {holding_period} Days [%]",
            side="right",
            overlaying="y",
            title_font=dict(color=st_dark_blue),
            hoverformat=".2f",
        ),
        yaxis3=dict(
            title="Proposed Leverage Factor",
            side="right",
            overlaying="y",
            position=0.96,
            title_font=dict(color=st_red),
            hoverformat=".2f",
        ),
        xaxis=xaxis_slider(price),
    )
    # update x width
    fig.update_xaxes(
        domain=[0.0, 0.89],
    )

    return fig


def update_derivates_calibration_plot(
    data_dict: dict[str, pd.Series | list | float], leverage: float, holding_period: int
) -> go.Figure:
    # create the plotly figure
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    # unpack the data dictionary
    name = data_dict["name"]
    returns_1x = data_dict["returns_1x"]
    returns_lev = data_dict["returns_lev"]
    returns_ko = data_dict["returns_ko"]
    opacities_lev = data_dict["opacities_lev"]
    opacities_ko = data_dict["opacities_ko"]

    # add violin plots to the first subplot
    fig.add_trace(
        go.Violin(
            y=returns_lev,
            name=f"{leverage}x Factor",
            box_visible=True,
            line_color=st_dark_blue,
            side="negative",
            points="all",
            pointpos=0.4,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Violin(
            y=returns_ko,
            name=f"{leverage}x Knockout",
            box_visible=True,
            line_color=st_darker_blue,
            side="negative",
            points="all",
            pointpos=0.4,
        ),
        row=1,
        col=1,
    )

    # add calibration plot to the second subplot
    fig.add_trace(
        go.Scatter(
            x=returns_1x,
            y=returns_lev,
            mode="markers",
            name=f"{leverage}x Factor",
            marker=dict(
                color=st_dark_blue, symbol="triangle-up", opacity=opacities_lev
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=returns_1x,
            y=returns_ko,
            mode="markers",
            name=f"{leverage}x Knockout",
            marker=dict(color=st_darker_blue, symbol="square", opacity=opacities_ko),
        ),
        row=1,
        col=2,
    )
    # add derivatives zero return line
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=0,
        y0=min(min(returns_lev), min(returns_ko)),
        x1=0,
        y1=max(max(returns_lev), max(returns_ko)),
        line=dict(
            color=st_green,
            width=2,
            dash="dash",
        ),
        row=1,
        col=2,
    )
    # add underlying zero returns line
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=min(returns_1x),
        y0=0,
        x1=max(returns_1x),
        y1=0,
        line=dict(
            color=st_darker_blue,
            width=2,
            dash="dash",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title="Underlying Returns [%]",
        title_font=dict(color=st_green),
        hoverformat=".2f",
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"<span style='font-size: 24px;'>{name} Derivative Performance Calibration</span><br>"
        + "<span style='font-size: 16px;'>Comparison of the Returns of the Underlying"
        + f" and {leverage}x Factors and Knockouts for {holding_period} Day Time Intervals</span>",
        hovermode="x unified",
        yaxis=dict(
            title=f"{leverage}x Factor and Knockout Returns [%]",
            title_font=dict(color=st_darker_blue),
            hoverformat=".2f",
        ),
    )

    return fig
