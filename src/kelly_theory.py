"""
Define the functions to update the Kelly Criterion heatmap and the LETF outperformance 
heatmap.

"""

import numpy as np
import plotly.graph_objects as go

from src import constants
from src.utils import kelly_crit, leveraged_return_mesh


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


# define heatmap marginals (-50% - 50% underlying CAGR, 0-50% annualized volatility)
cagr_underlying = np.linspace(-50, 50, 201, endpoint=True)
volatility_undr = np.linspace(0.0, 50, 101, endpoint=True)


# define the data
zmin, zmax = -15, 20
data = [
    go.Heatmap(
        x=cagr_underlying,
        y=volatility_undr,
        z=leveraged_return_mesh(
            constants.lev_r,
            cagr_underlying,
            constants.exp_r,
            constants.libor,
            volatility_undr,
        ),
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
        z=leveraged_return_mesh(
            constants.lev_r,
            cagr_underlying,
            constants.exp_r,
            constants.libor,
            volatility_undr,
        ),
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


# define heatmap marginals (-50% - 50% underlying CAGR, 0-50% annualized volatility)
cagr_f_underlying = np.linspace(-20, 60, 161, endpoint=True)
volatility_f_undr = np.linspace(0.5, 50, 100, endpoint=True)


# define the Kelly criterion data
zmin_f, zmax_f = 0, 10
data_f = [
    go.Heatmap(
        x=cagr_f_underlying,
        y=volatility_f_undr,
        z=kelly_crit_mesh(
            cagr_f_underlying, constants.ann_risk_free, volatility_f_undr
        ),
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
        z=kelly_crit_mesh(
            cagr_f_underlying, constants.ann_risk_free, volatility_f_undr
        ),
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
