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
import plotly.graph_objects as go
import streamlit as st

from utils import fetch_ticker_data, leveraged_return_mesh


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
    fig = go.Figure()  # make_subplots(specs=[[{"secondary_y": True}]])
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
    # add price line
    fig.add_trace(
        go.Scatter(
            x=result_dict["price"].index[-252:],
            y=result_dict["price"].iloc[-252:],
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
    # add volatility
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
    # add daily volatility
    fig.add_trace(
        go.Scatter(
            x=result_dict["volatility"].index,
            y=result_dict["volatility"],
            mode="lines",
            name="Daily Volatility Estimate",
            yaxis="y2",
            visible="legendonly",
            line=dict(color=st_red, dash="dot"),
        )
    )
    # add garch volatility
    fig.add_trace(
        go.Scatter(
            x=result_dict["garch_volatility"].index,
            y=result_dict["garch_volatility"],
            mode="lines",
            name="GARCH Forecast Volatility",
            yaxis="y2",
            visible="legendonly",
            line=dict(color=st_light_red, dash="dash"),
        )
    )
    # add daily percentage change
    pct_change = result_dict["price"].pct_change() * 100
    fig.add_trace(
        go.Scatter(
            x=result_dict["price"].index[-252:],
            y=pct_change.iloc[-252:],
            mode="lines",
            name="Daily Returns",
            yaxis="y3",
            visible="legendonly",
            line=dict(color=st_green),
        )
    )

    # calculate the Kelly Criterion
    annual_return = result_dict["price"].iloc[-1] / result_dict["price"].iloc[-252] - 1
    kelly = kelly_crit(
        annual_return * 100,
        risk_free_rate_ticker,
        result_dict["ann_volatility"].iloc[-1],
    )
    # calculate the leverage factor for 20% volatility
    lev_20 = 20 / result_dict["ann_volatility"].iloc[-1]
    # calculate the Percentage at Risk (PaR)
    par_5 = np.percentile(pct_change.dropna(), 5)
    par_1 = np.percentile(pct_change.dropna(), 1)

    # update layout
    fig.update_layout(
        title=f"<span style='font-size: 24px;'>Current Price and Volatility of {result_dict['name']}</span><br>"
        + "<span style='font-size: 4px;'></span><br>"
        + f"<span style='font-size: 16px;'>Suggested Kelly Leverage Factor: {kelly:.2f}\
   -   Leverage @20% Volatility: {lev_20:.2f}\
   -   Current 1%/5% PaR (last 5y): {par_1:.2f}%/{par_5:.2f}% of the Underlying</span>",
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
    )
    # set x-ticks
    num_ticks = 10
    tickids = np.linspace(
        0, min(252, len(result_dict["price"])) - 1, num_ticks, endpoint=True, dtype=int
    )
    tickvals = [result_dict["price"].iloc[-252:].index[id] for id in tickids]
    ticktext = [val.strftime("%Y-%m-%d") for val in tickvals]
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        domain=[0.0, 0.89],
    )

    return fig


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

    # Header for the Kelly Criterion
    st.markdown("", unsafe_allow_html=True)
    st.write(
        "## [Kelly Criterion](https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/) Calculator (Annualized)"
    )
    # Text input for yearly expected return
    input_er = st.number_input("Expected Yearly Return [%]", value=ann_return)
    # Text input for yearly risk free rate
    input_risk_free = st.number_input(
        "Risk Free Yearly Return [%]", value=ann_risk_free
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
        "Risk Free Yearly Return [%]",
        min_value=0.0,
        max_value=8.0,
        value=3.0,
        step=0.25,
    )
    # Placeholder for the graph
    st.plotly_chart(
        update_kelly_plot(data_source_kelly, risk_free_rate), use_container_width=True
    )
    # Header for the ticker price data plot
    st.markdown("", unsafe_allow_html=True)
    st.write("## What Does All This Imply for My Investments?")
    st.markdown(
        """
        Depending on whether you want to pursue a maximally aggressive Kelly strategy 
        or whether you are aiming for an annualized volatility of 20%, for example, 
        you can find suggestions below in view of current stock market prices as well 
        as forecasts for the percentage-at-risk (PaR) at empirical 1% and 5% levels 
        based on the data of the last 5 years (if available) in order to put the Kelly 
        suggestions into perspective.\n
        
        Here are some common ticker symbols:<br>
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
        "Risk Free Yearly Return [%]",
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
