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

from utils import leveraged_return_mesh

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
