"""
This module contains the code for the volatility decay dashboard.

Required libraries:
pip install dash dash-core-components dash-html-components dash-renderer

Based on a [reddit](https://www.reddit.com/r/HFEA/comments/tue7n6/the_volatility_decay_equation_with_verification/) 
post, make an interactive visualization to show the effect of the volatility decay.

The results show the (somewhat) quadratic (/ logarithmic) volatility drag along the volatility axis, together with 
(somewhat) quadratic (/ logarithmic) scaling profit region decrease with increased leverage factor. Further sources 
describing the quadratic behaviour:
- [Blogpost](https://www.afrugaldoctor.com/home/leveraged-etfs-and-volatility-decay-part-2)
- [(Detailed) Journal Article, also mentioned in the Blogpost](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1664823)

"""

import numpy as np
import plotly.graph_objects as go

import streamlit as st


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

    return mesh * 100


# define display functions
def update_result(a, b, c):
    # calculate the Kelly Criterion
    # NOTE: the factor of 100 corrects for the percentage values
    # NOTE: the factor of 252 corrects for the daily values
    return f"#### Kelly Fraction f: {100 * (a - b) / c**2:.2f}"

# define parameters (except for leverage all in percent)
lev_r = 2.0
exp_r = 0.6
libor = 0.5
# define heatmap marginals (-50% - 50% underlying CAGR, 0-100% annualized volatility)
cagr_underlying = np.linspace(-50, 50, 200)
volatility_undr = np.linspace(0.0, 100, 100)

# define the data
data = [
    go.Heatmap(
        x=cagr_underlying,
        y=volatility_undr,
        z=leveraged_return_mesh(lev_r, cagr_underlying, exp_r, libor, volatility_undr),
        zmax=20,
        zmid=0,
        zmin=-15,
        colorscale="RdBu",
        colorbar=dict(title="Gain over Unleveraged ETF [%]", titleside="right"),
    )
]
data_contour = [
    go.Contour(
        x=cagr_underlying,
        y=volatility_undr,
        z=leveraged_return_mesh(lev_r, cagr_underlying, exp_r, libor, volatility_undr),
        zmax=20,
        zmid=0,
        zmin=-15,
        colorscale="RdBu",
        colorbar=dict(title="Gain over Unleveraged ETF [%]", titleside="right"),
    )
]

def update_plot(data_source, leverage, TER, LIBOR):
    # update data
    if data_source == "heatmap":
        fig = go.FigureWidget(data=data)
    else:
        fig = go.FigureWidget(data=data_contour)
    # write data to figure
    fig.data[0].z = leveraged_return_mesh(
        leverage, cagr_underlying, TER, LIBOR, volatility_undr
    )
    # update layout
    fig.update_layout(
        title="Visualized Gain over Unleveraged ETF",
        title_x=0.35,  # Center the title
        title_font=dict(size=24),
        xaxis_title="CAGR Underlying [%]",
        yaxis_title="Volatility [%]",
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
            color: white;
            max-width: 100%;
            text-align: center;
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
    st.write("## Kelly Criterion Calculator")
    # Text input for yearly expected return
    yearly_er = st.number_input("Yearly Expected Return [%]", value=0.037 * 252)
    # Text input for yearly risk free rate
    yearly_risk_free = st.number_input("Yearly Risk Free Rate", value=0.005 * 252)
    # Text input for yearly return volatility
    yearly_volatility = st.number_input("Yearly Return Volatility [%]", value=1.2 * np.sqrt(252))
    # Display the result
    result = update_result(yearly_er, yearly_risk_free, yearly_volatility)
    st.write(result)

    # Header for the plot
    st.markdown("", unsafe_allow_html=True)
    st.write("## Visualize the Gain over Unleveraged ETF")
    # Dropdown for the plot style
    data_source = st.selectbox("Plot Style", ["Heatmap", "Contour"], index=0)
    # Slider for leverage
    leverage = st.slider("Leverage", min_value=0.0, max_value=10.0, value=lev_r, step=0.5)
    # Slider for TER
    ter = st.slider("TER", min_value=0.0, max_value=1.0, value=exp_r, step=0.05)
    # Slider for LIBOR
    libor = st.slider("LIBOR", min_value=0.0, max_value=3.0, value=libor, step=0.25)
    # Placeholder for the graph
    st.plotly_chart(update_plot(data_source, leverage, ter, libor), use_container_width=True)
