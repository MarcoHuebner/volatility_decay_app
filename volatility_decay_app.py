"""
This module contains the code for the volatility decay Streamlit App.

Based on a [reddit](https://www.reddit.com/r/HFEA/comments/tue7n6/the_volatility_decay_equation_with_verification/) 
post, make an interactive visualization to show the effect of the volatility decay.

The results show the (somewhat) quadratic (/ logarithmic) volatility drag along the volatility axis, together with 
(somewhat) quadratic (/ logarithmic) scaling profit region decrease with increased leverage factor. Further sources 
describing the quadratic behaviour:
- [Blogpost](https://www.afrugaldoctor.com/home/leveraged-etfs-and-volatility-decay-part-2)
- [(Detailed) Journal Article, also mentioned in the Blogpost](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1664823)

requirements.txt:
plotly
numpy
streamlit
ipywidgets>=7.0.0

"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    fetch_ticker_data,
    kelly_leverage,
    leveraged_return_mesh,
    performance_cumprod,
    plot_earnings_dates,
    simplified_knockout,
    simplified_lev_factor,
    xaxis_slider,
)

# define display functions
ann_return = 0.037 * 252
ann_risk_free = 0.005 * 252
ann_vol = 1.2 * np.sqrt(252)
# define heatmap marginals (-50% - 50% underlying CAGR, 0-50% annualized volatility)
cagr_f_underlying = np.linspace(-20, 60, 161, endpoint=True)
volatility_f_undr = np.linspace(0.5, 50, 100, endpoint=True)


def kelly_crit(
    yearly_er: float, yearly_risk_free: float, yearly_volatility: float
) -> float:
    # calculate the Kelly Criterion
    # NOTE: the factor of 100 corrects for the percentage values
    assert np.all(yearly_volatility > 0)
    return 100 * (yearly_er - yearly_risk_free) / yearly_volatility**2


def update_result(
    yearly_er: float, yearly_risk_free: float, yearly_volatility: float
) -> str:
    # display the Kelly Criterion
    kelly_f = kelly_crit(yearly_er, yearly_risk_free, yearly_volatility)
    return f"#### Ideal Market Exposure: {kelly_f*100:.0f}% ({kelly_f:.2f} leverage factor)"


def kelly_crit_mesh(
    yearly_er: float, yearly_risk_free: float, yearly_volatility: float
) -> np.ndarray:
    # calculate the Kelly Criterion meshed for different underlying CAGR and volatility
    mesh = np.zeros((len(yearly_volatility), len(yearly_er)))
    for i, vol in enumerate(yearly_volatility):
        for j, cagr in enumerate(yearly_er):
            # reflect on volatility axis due to the way, plotly sets-up heatmaps
            # also, rescale percentage values, as otherwise not readable in the sliders
            mesh[i, j] = kelly_crit(cagr, yearly_risk_free, vol)
    return np.round(mesh, 2)


# define parameters (except for leverage all in percent)
lev_r = 2.0
exp_r = 0.6
libor = 0.5
# define heatmap marginals (-50% - 50% underlying CAGR, 0-50% annualized volatility)
cagr_underlying = np.linspace(-50, 50, 201, endpoint=True)
volatility_undr = np.linspace(0.0, 50, 101, endpoint=True)

# define the data
zmin, zmax = -15, 20
data = [
    go.Heatmap(
        x=cagr_underlying,
        y=volatility_undr,
        z=leveraged_return_mesh(lev_r, cagr_underlying, exp_r, libor, volatility_undr),
        zmax=zmax,
        zmid=0,
        zmin=zmin,
        colorscale="RdBu",
        colorbar=dict(
            title="Outperformance LETF over Unleveraged ETF [%]", titleside="right"
        ),
    )
]
data_contour = [
    go.Contour(
        x=cagr_underlying,
        y=volatility_undr,
        z=leveraged_return_mesh(lev_r, cagr_underlying, exp_r, libor, volatility_undr),
        zmax=zmax,
        zmid=0,
        zmin=zmin,
        colorscale="RdBu",
        colorbar=dict(
            title="Outperformance LETF over Unleveraged ETF [%]", titleside="right"
        ),
    )
]


def update_plot(
    data_source: str, leverage: float, TER: float, LIBOR: float
) -> go.Figure:
    # rescale the leverage factor
    leverage /= 100
    # update data
    if data_source == "Heatmap":
        fig = go.FigureWidget(data=data)
    else:
        fig = go.FigureWidget(data=data_contour)
    # write data to figure
    fig.data[0].z = leveraged_return_mesh(
        leverage, cagr_underlying, TER, LIBOR, volatility_undr
    )
    # update layout
    fig.update_layout(
        title="Gain of the LETF over the Unleveraged ETF",
        title_font=dict(size=24),
        hovermode="x unified",
        xaxis_title="Underlying: Expected Yearly Return [%]",
        yaxis_title="Underlying: Annualized Volatility [%]",
    )
    fig.update_traces(
        hovertemplate="Outperformance [%]: %{z:.1f}<br>Underlying: CAGR [%]: %{x}<br>Underlying: Ann. Vol. [%]: %{y}<extra></extra>"
    )

    return fig


# define the Kelly criterion data
zmin_f, zmax_f = 0, 10
data_f = [
    go.Heatmap(
        x=cagr_f_underlying,
        y=volatility_f_undr,
        z=kelly_crit_mesh(cagr_f_underlying, ann_risk_free, volatility_f_undr),
        zmin=zmin_f,
        zmid=1,
        zmax=zmax_f,
        colorscale="RdylGn_r",
        colorbar=dict(title="Ideal Leverage Factor", titleside="right"),
    )
]
data_f_contour = [
    go.Contour(
        x=cagr_f_underlying,
        y=volatility_f_undr,
        z=kelly_crit_mesh(cagr_f_underlying, ann_risk_free, volatility_f_undr),
        zmin=zmin_f,
        zmid=1,
        zmax=zmax_f,
        colorscale="RdylGn_r",
        colorbar=dict(title="Ideal Leverage Factor", titleside="right"),
    )
]


def update_kelly_plot(data_source: str, risk_free_rate: float) -> go.Figure:
    # update data
    if data_source == "Heatmap":
        fig = go.FigureWidget(data=data_f)
    else:
        fig = go.FigureWidget(data=data_f_contour)
    # write data to figure
    fig.data[0].z = kelly_crit_mesh(
        cagr_f_underlying, risk_free_rate, volatility_f_undr
    )
    # update layout
    fig.update_layout(
        title="Ideal Leverage Factor According to Kelly Criterion",
        title_font=dict(size=24),
        hovermode="x unified",
        xaxis_title="Underlying: Expected Yearly Return [%]",
        yaxis_title="Underlying: Annualized Volatility [%]",
    )
    fig.update_traces(
        hovertemplate="Leverage Factor: %{z}<br>Underlying: CAGR [%]: %{x}<br>Underlying: Ann. Vol. [%]: %{y}<extra></extra>"
    )
    # set initial zoom
    fig.update_xaxes(range=[-5, 15])
    fig.update_yaxes(range=[0.5, 25])

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
        + f"Volatility of {result_dict['name']}</span><br>"
        + f"<span style='font-size: 16px;'>Kelly Leverage Factor: {kelly:.2f}"
        + f" - Leverage @20% Volatility: {lev_20:.2f}"
        + f" - Current 1%/5% PaR (last 2y): {par_1:.2f}%/{par_5:.2f}%</span>",
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
    # add last holding_period day cut-off line
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


if __name__ == "__main__":
    # define the layout
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .header {
            background: linear-gradient(to right, #67001f, #053061);
            padding: 2em;
            color: white !important;
            max-width: 100%;
            text-align: center;
        }
        .header h1, .header h3 {
            color: white !important;
        }
        </style>

        <div class="header">
            <h1>Volatility Decay & Kelly Criterion</h1>
            <h3>Volatility decay, the phenomenon of underperforming the market 
            despite having a larger position, is a central concept in the realm 
            of financial markets and risk management. Navigating this environment 
            requires strategic and rational decision-making when it comes to 
            position sizing, and the Kelly Criterion, developed by John L. Kelly Jr,
            is proving to be a valuable tool. This formula allows investors to optimize
            the size of their positions while balancing long-term growth expectations with
            risk mitigation. By incorporating probabilities of success and risk/return
            ratios, the Kelly Criterion provides a smart strategy for investors looking
            to make informed decisions in the stock market.</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Kelly Theory", "Investments", "Stock Universe Screener (WIP)"]
    )

    with tab1:
        # Header for the Kelly Criterion
        st.markdown("", unsafe_allow_html=True)
        st.write(
            "## [Kelly Criterion](https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/) Calculator (Annualized)"
        )
        # Text input for yearly expected return
        input_er = st.number_input("Expected Yearly Return [%]", value=ann_return)
        # Text input for yearly risk free rate
        input_risk_free = st.number_input(
            "Risk Free Yearly Return [%] (Costs)", value=ann_risk_free
        )
        # Text input for yearly return volatility
        input_volatility = st.number_input(
            "Annualized Volatility of the Underlying [%]", value=ann_vol
        )
        # Display the result
        result = update_result(input_er, input_risk_free, input_volatility)
        st.write(result)

        # Header for the plot
        st.markdown("", unsafe_allow_html=True)
        st.write("## Gain of the LETF over the Unleveraged ETF")
        st.markdown(
            """
            The difference in profit of the hypothetical leveraged ETF is 
            compared to the unleveraged ETF below. It becomes clear that 
            the profit does not increase linearly with leverage and the 
            margin of error becomes smaller and smaller, especially with 
            higher leverage.
            """
        )
        # Dropdown for the plot style
        data_source = st.selectbox("Plot Style", ["Heatmap", "Contour"], index=1)
        # Slider for leverage
        leverage = st.slider(
            "Exposure to the Market (Leverage) [%]",
            min_value=0.0,
            max_value=1000.0,
            value=lev_r * 100,
            step=10.0,
        )
        # Slider for TER
        ter = st.slider(
            "TER of the ETF [%]", min_value=0.0, max_value=2.0, value=exp_r, step=0.05
        )
        # Slider for LIBOR
        libor = st.slider(
            "LIBOR rate [%]", min_value=0.0, max_value=4.0, value=libor, step=0.1
        )
        # Placeholder for the graph
        st.plotly_chart(
            update_plot(data_source, leverage, ter, libor), use_container_width=True
        )

        # Header for the Kelly Criterion plot
        st.markdown("", unsafe_allow_html=True)
        st.write("## Ideal Market Exposure (Leverage) According to the Kelly Criterion")
        st.markdown(
            """
            In contrast to the previous figure, the ideal leverage factor, which 
            is determined by the Kelly criterion, shows an even stronger dependence 
            on volatility and thus an even smaller margin of error. This is due 
            to the fact that the Kelly criterion maximizes the expected geometric 
            growth rate, thus avoiding the 
            ['Just One More Paradox'](https://www.youtube.com/watch?v=_FuuYSM7yOo&), 
            which describes the phenomenon of a median loss even though the expected 
            value of every bet and thus the overall expected value (mean) is positive.
            """,
            unsafe_allow_html=True,
        )
        # Dropdown for the plot style
        data_source_kelly = st.selectbox(
            "Plot Style", ["Heatmap", "Contour"], index=1, key="kelly"
        )
        # Slider for the risk free rate
        risk_free_rate = st.slider(
            "Risk Free Yearly Return [%] (Costs)",
            min_value=0.0,
            max_value=8.0,
            value=3.0,
            step=0.25,
        )
        # Placeholder for the graph
        st.plotly_chart(
            update_kelly_plot(data_source_kelly, risk_free_rate),
            use_container_width=True,
        )

    with tab2:
        # Header for the ticker price data plot
        st.markdown("", unsafe_allow_html=True)
        st.write("## What Does All This Imply for My Investments?")
        st.markdown(
            """
            Depending on whether you want to pursue a maximally aggressive Kelly strategy 
            or whether you are aiming for an annualized volatility of 20%, for example, 
            you can find suggestions below in view of current stock market prices as well 
            as forecasts for the percentage-at-risk (PaR) at empirical 1% and 5% levels 
            based on the data of the last 2 years (if available) in order to put the Kelly 
            suggestions into perspective.\n
            Keep in mind that an overestimation of returns and and an underestimation of
            volatility can lead to an inflated allocation fraction and thus a significant 
            loss of capital <a href="#footnote-1">[1]</a>. The Kelly Criterion is a 
            powerful tool, but it is not a guarantee for success. It is important to use 
            it in conjunction with other risk management strategies and to be aware of the 
            limitations of the model and its assumptions.\n
            
            Further Reading:
            <p id="footnote-1">[1] E. Thorp, <a href=
            "https://www.eecs.harvard.edu/cs286r/courses/fall12/papers/Thorpe_KellyCriterion2007.pdf">
            THE KELLY CRITERION IN BLACKJACK SPORTS BETTING, AND THE STOCK MARKET</a>, 
            Chapter 9, Handbook of Asset and Liability Management.</p>
            
            For the practical part, here are some common ticker symbols:<br>
            - ^GSPC: S&P 500 Index<br>
            - ^NDX: Nasdaq 100 Index<br>
            - AAPL: Apple Inc.<br>
            - MSFT: Microsoft Corporation<br>
            - NVDA: NVIDIA Corporation<br>
            - ^GDAXI: DAX40 Index<br>
            - ^STOXX50E: EURO STOXX 50 Index<br>
            - ^STOXX: STOXX Europe 600 Index<br>
            - ^N225: Nikkei 225 Index<br>
            - SPY: SPDR S&P 500 ETF<br>
            - SJIM: Inverse Cramer ETF\n
            
            For a list of all available ticker symbols, please visit 
            https://finance.yahoo.com/lookup or https://finance.yahoo.com/most-active/
            """,
            unsafe_allow_html=True,
        )
        # Ticker string input
        ticker_symbol = st.text_input("Ticker Symbol", value="^GSPC")
        # Slider for the risk free rate
        risk_free_rate_ticker = st.slider(
            "Risk Free Yearly Return [%] (Costs)",
            min_value=0.0,
            max_value=8.0,
            value=3.0,
            step=0.25,
            key="ticker",
        )
        st.plotly_chart(
            update_ticker_plot(ticker_symbol, risk_free_rate_ticker),
            use_container_width=True,
        )
        # Slider for the expenses of the derivatives
        derivative_expenses = st.slider(
            "Yearly Expense Ratio of the Derivatives [%]",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.25,
        )
        # Slider for the transaction costs of the derivatives
        rel_transact_costs = st.slider(
            "Transaction Costs for Buying and Selling (Separately) [%]",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.25,
        )
        # Slider for the Kelly leverage time window
        look_back_window = st.slider(
            "Kelly Leverage Look-Back Window [Trading Days]",
            min_value=10,
            max_value=100,
            value=60,
            step=10,
            format="%d",
        )
        # Slider for the holding period of the derivatives
        holding_period = st.slider(
            "Holding Period of the Derivatives [Trading Days]",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            format="%d",
        )
        # Slider for the leverage of the derivatives
        derivative_leverage = st.slider(
            "Leverage of the Derivatives",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.25,
        )
        (
            derivates_fig,
            win_ratio_ko,
            win_ratio_f,
            reward_ko,
            reward_f,
        ) = update_derivatives_performance_plot(
            ticker_symbol,
            risk_free_rate_ticker,
            derivative_leverage,
            derivative_expenses,
            rel_transact_costs,
            look_back_window,
            holding_period,
        )
        st.write(
            f"Win ratio over the past {holding_period} trading days"
            + f" (KO / Factor): {win_ratio_ko:.1f}%/ {win_ratio_f:.1f}%."
            + " Average risk (loss) vs. average reward (win) per trade"
            + f" over the past {holding_period} trading days (KO/ Factor):"
            + f" 1:{reward_ko:.2f}/ 1:{reward_f:.2f}"
        )
        st.plotly_chart(
            derivates_fig,
            use_container_width=True,
        )

    with tab3:
        # Header for the Stock Screener
        st.markdown("", unsafe_allow_html=True)
        st.write("## Work in Progress")
