"""
This module contains the code for the volatility decay dashboard.

Required libraries:
pip install dash dash-core-components dash-html-components dash-renderer

assets folder: Contains style.css file for styling the dashboard

Based on a [reddit](https://www.reddit.com/r/HFEA/comments/tue7n6/the_volatility_decay_equation_with_verification/) 
post, make an interactive visualization to show the effect of the volatility decay.

The results show the (somewhat) quadratic (/ logarithmic) volatility drag along the volatility axis, together with 
(somewhat) quadratic (/ logarithmic) scaling profit region decrease with increased leverage factor. Further sources 
describing the quadratic behaviour:
- [Blogpost](https://www.afrugaldoctor.com/home/leveraged-etfs-and-volatility-decay-part-2)
- [(Detailed) Journal Article, also mentioned in the Blogpost](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1664823)

"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output


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


# initialize the app
app = dash.Dash(__name__)
# define parameters (except for leverage all in percent)
lev_r = 2
exp_r = 0.6
libor = 0.5
# define heatmap marginals (-50% - 50% underlying CAGR, 0-100% annualized volatility)
cagr_underlying = np.linspace(-50, 50, 200)
volatility_undr = np.linspace(0.0, 100, 100)

# define the layout
app.layout = html.Div(
    [
        # define the header
        html.Header(
            children=[
                html.H1(
                    children="Volatility Decay & Kelly Criterion",
                    style={
                        "textAlign": "center",
                        "fontSize": "40px",
                        "color": "#ffffff",
                    },
                ),
                html.H3(
                    children="Volatility decay, the phenomenon of underperforming the market "
                    + "despite having a larger position, is a central concept in the realm "
                    + "of financial markets and risk management. Navigating this environment "
                    + "requires strategic and rational decision-making when it comes to "
                    + "position sizing, and the Kelly Criterion, developed by John L. Kelly Jr, "
                    + "is proving to be a valuable tool. This formula allows investors to optimize "
                    + "the size of their positions while balancing long-term growth expectations with "
                    + "risk mitigation. By incorporating probabilities of success and risk/return "
                    + "ratios, the Kelly Criterion provides a smart strategy for investors looking "
                    + "to make informed decisions in the stock market.",
                    style={
                        "textAlign": "center",
                        "width": "70vw",
                        "fontSize": "20px",
                        "color": "#ffffff",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "alignItems": "center",
                "textAlign": "center",
                "height": "35vh",
                "margin": "0",
                "fontSize": "40px",
                "color": "#ffffff",
                "background": "linear-gradient(to right, #67001f, #053061)",
            },
        ),
        # TODO: Make auto-scrolling directly go all the way to the bottom
        html.Div(id="page-content"),
        # add a description
        # TODO: Add a more detailed description and Kelly Plot (conservative -> danger)
        html.Div(
            [
                html.Span(
                    "Calculating the ",
                    style={"fontSize": "1.5em", "fontWeight": "bold"},
                ),
                html.A(
                    "Kelly Criterion",
                    href="https://rhsfinancial.com/2017/06/20/line-aggressive-crazy-leverage/",
                    style={"fontSize": "1.5em", "fontWeight": "bold"},
                    target="_blank",
                ),
            ],
            style={"textAlign": "center", "margin": "5vh auto auto auto"},
        ),
        # add sliders to calculate the Kelly Criterion
        html.Div(
            [
                html.Label(
                    "Yearly Expected Return [%]",
                    style={"fontSize": "20px", "marginRight": "1vw"},
                ),
                dcc.Input(id="yearly_er", type="number", value=0.037 * 252),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        html.Div(
            [
                html.Label(
                    "Yearly Risk Free Rate",
                    style={"fontSize": "20px", "marginRight": "1vw"},
                ),
                dcc.Input(id="yearly_risk_free", type="number", value=0.005 * 252),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        html.Div(
            [
                html.Label(
                    "Yearly Return Volatility [%]",
                    style={"fontSize": "20px", "marginRight": "1vw"},
                ),
                dcc.Input(
                    id="yearly_volatility",
                    type="number",
                    value=round(1.2 * np.sqrt(252), 2),
                ),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        html.Div(
            id="result",
            style={
                "textAlign": "center",
                "fontSize": "20px",
                "margin": "5vh auto auto auto",
            },
        ),
        # add a description
        html.H2(
            "Visualizing the Volatility Decay",
            style={"textAlign": "center", "margin": "5vh auto auto auto"},
        ),
        # add dropdown for the contour/ heatmap
        dcc.Dropdown(
            id="data-source",
            options=[
                {"label": "Plot Style: Heatmap", "value": "heatmap"},
                {"label": "Plot Style: Contour", "value": "contour"},
            ],
            value="heatmap",
            style={
                "textAlign": "center",
                "width": "90vw",
                "margin": "2vh auto auto auto",
            },
        ),
        # add sliders with headings for the parameters
        # TODO: Add a self-explaining description for each slider
        html.Div(
            [
                html.Label("Leverage", style={"fontSize": "20px"}),
                dcc.Slider(id="leverage-slider", min=0, max=10, step=0.5, value=lev_r),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        html.Div(
            [
                html.Label("TER", style={"fontSize": "20px"}),
                dcc.Slider(id="TER-slider", min=0, max=1, step=0.05, value=exp_r),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        html.Div(
            [
                html.Label("LIBOR", style={"fontSize": "20px"}),
                dcc.Slider(id="LIBOR-slider", min=0, max=3, step=0.25, value=libor),
            ],
            style={"textAlign": "center", "margin": "1vh auto 1vh auto"},
        ),
        # add the graph
        dcc.Graph(
            id="3d-plot",
            style={"width": "90vw", "height": "65vh", "margin": "auto"},
        ),
    ]
)

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


# define the callback for the calculation
@app.callback(
    Output("result", "children"),
    [
        Input("yearly_er", "value"),
        Input("yearly_risk_free", "value"),
        Input("yearly_volatility", "value"),
    ],
)
def update_result(a, b, c):
    # calculate the Kelly Criterion
    # NOTE: the factor of 100 corrects for the percentage values
    # NOTE: the factor of 252 corrects for the daily values
    return f"Kelly Fraction f: {100 * (a - b) / c**2:.2f}"


# define the callback for the plot
@app.callback(
    Output("3d-plot", "figure"),
    [
        Input("data-source", "value"),
        Input("leverage-slider", "value"),
        Input("TER-slider", "value"),
        Input("LIBOR-slider", "value"),
    ],
)
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
        title_x=0.5,  # Center the title
        title_font=dict(size=24),
        xaxis_title="CAGR Underlying [%]",
        yaxis_title="Volatility [%]",
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
