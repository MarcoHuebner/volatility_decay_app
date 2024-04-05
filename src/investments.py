"""
Define the functions to update the plots for the investments page.

"""

import numpy as np
import plotly.graph_objects as go

from src.utils import (
    fetch_ticker_data,
    kelly_crit,
    kelly_leverage,
    performance_cumprod,
    plot_earnings_dates,
    simplified_knockout,
    simplified_lev_factor,
    xaxis_slider,
)


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
    price = result_dict["price"].tail(252)
    # get earning dates if not None (e.g. for indices)
    if result_dict["earnings"] is not None:
        earnings = result_dict["earnings"]["Reported EPS"]
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
    pct_change = result_dict["price"].pct_change() * 100
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=pct_change.iloc[-252:],
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
    average_daily_return = pct_change.iloc[-252:].mean()
    max_vol = max(
        result_dict["ann_volatility"].iloc[-1],
        # result_dict["garch_volatility"].iloc[-1],
        average_vol_30d,
    )
    # add a safety margin of -2% ann. return and +3% ann. volatility
    safety_return = -2
    safety_vol = 3
    kelly = kelly_crit(
        average_daily_return * 252 + safety_return,
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


# define the past leverage and knock out returns plot
def update_derivatives_performance_plot(
    ticker: str,
    risk_free_rate_ticker: float,
    leverage: float,
    expenses: float,
    rel_transact_costs: float,
    time_window: int,
    holding_period: int,
) -> go.Figure:
    # create the plotly figure
    fig = go.Figure()
    # define colours, loosely related to the streamlit default colours
    # https://discuss.streamlit.io/t/access-streamlit-default-color-palette/35737
    st_blue = "#83c9ff"
    st_dark_blue = "#0068c9"
    st_darker_blue = "#0054a3"
    st_red = "#ff2b2b"
    st_green = "#21c354"

    # get data
    result_dict = fetch_ticker_data(ticker)
    price = result_dict["price"].tail(252)
    pct_change = result_dict["price"].pct_change()
    # get earning dates if not None (e.g. for indices)
    if result_dict["earnings"] is not None:
        earnings = result_dict["earnings"]["Reported EPS"]

    # how have derivatives bought with kelly criterion > 5 performed in the past?
    # show results of holding_period day intervals
    kelly_crit = kelly_leverage(
        pct_change, risk_free_rate_ticker, time_window=time_window
    ).tail(252)
    pct_change = pct_change.tail(252)
    # get days on which the kelly criterion was > 5
    dates_iloc = np.where(kelly_crit > 5)[0]
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
                    expenses,
                    rel_transact_costs,
                    holding_period,
                    leverage,
                )
            )
            for date in dates_iloc
        ]
        # Pre-compute quantities to simplify the code
        pos_returns_ko = [r for r in returns_ko if r > 0]
        avg_win_ko = sum(pos_returns_ko) / len(pos_returns_ko)
        neg_returns_ko = [r for r in returns_ko if r <= 0]
        avg_loss_ko = sum(neg_returns_ko) / len(neg_returns_ko)
        pos_returns_f = [r for r in returns_lev if r > 0]
        avg_win_f = sum(pos_returns_f) / len(pos_returns_f)
        neg_returns_f = [r for r in returns_lev if r <= 0]
        avg_loss_f = sum(neg_returns_f) / len(neg_returns_f)
        # Calculate win ratios and reward ratios
        win_ratio_ko = len(pos_returns_ko) / len(returns_ko) * 100
        win_ratio_f = len(pos_returns_f) / len(returns_lev) * 100
        reward_ko = avg_win_ko / abs(avg_loss_ko)
        reward_f = avg_win_f / abs(avg_loss_f)
    # Calculate opacities based on comparison of returns
    opacities_lev = [
        0.3 if lev <= ko else 1.0 for lev, ko in zip(returns_lev, returns_ko)
    ]
    opacities_ko = [
        0.3 if ko < lev else 1.0 for lev, ko in zip(returns_lev, returns_ko)
    ]

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
    if result_dict["earnings"] is not None:
        # add earnings dates
        fig = plot_earnings_dates(earnings, price, fig)
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
            x=kelly_crit.index,
            y=kelly_crit,
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
            x=[kelly_crit.index.min(), kelly_crit.index.max()],
            y=[5, 5],
            mode="lines",
            yaxis="y3",
            line=dict(color=st_red, dash="dash"),
            visible="legendonly",
            legendgroup="group1",
            showlegend=False,
        )
    )

    # update layout
    fig.update_layout(
        title="<span style='font-size: 24px;'>How Derivatives of "
        + f"{result_dict['name']} Performed</span><br>"
        + "<span style='font-size: 16px;'>Given a 60-day Rolling Kelly "
        + "Leverage Factor as Signal - Amount of Signals in the Past 252 "
        + f"Trading Days: {len(dates_iloc)}</span>",
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

    return fig, win_ratio_ko, win_ratio_f, reward_ko, reward_f
