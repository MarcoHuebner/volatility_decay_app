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
ipywidgets>=7.0.0
numpy
plotly
prophet
streamlit

"""

import streamlit as st

from src import constants
from src.investments import (
    get_derivatives_data,
    update_derivates_calibration_plot,
    update_derivatives_performance_plot,
    update_ticker_plot,
)
from src.kelly_theory import update_kelly_plot, update_plot, update_result
from src.stock_screener import (
    compute_adr,
    kelly_selection,
    positive_return_selection,
    volatility_selection,
)
from src.utils import download_universe

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
        input_er = st.number_input(
            "Expected Yearly Return [%]", value=constants.ann_return
        )
        # Text input for yearly risk free rate
        input_risk_free = st.number_input(
            "Risk Free Yearly Return [%] (Costs)", value=constants.ann_risk_free
        )
        # Text input for yearly return volatility
        input_volatility = st.number_input(
            "Annualized Volatility of the Underlying [%]", value=constants.ann_vol
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
            value=constants.lev_r * 100,
            step=10.0,
        )
        # Slider for TER
        ter = st.slider(
            "TER of the ETF [%]",
            min_value=0.0,
            max_value=2.0,
            value=constants.exp_r,
            step=0.05,
        )
        # Slider for LIBOR
        libor = st.slider(
            "LIBOR rate [%]",
            min_value=0.0,
            max_value=4.0,
            value=constants.libor,
            step=0.1,
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
            based on the data of the past 2 years (if available) in order to put the Kelly 
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
        # Calculate the performance of the derivatives
        data_dict = get_derivatives_data(
            ticker_symbol,
            risk_free_rate_ticker,
            derivative_leverage,
            derivative_expenses,
            rel_transact_costs,
            look_back_window,
            holding_period,
        )
        # Display aggregated statistics of the derivatives
        win_ratio_ko = data_dict["win_ratio_ko"]
        win_ratio_f = data_dict["win_ratio_f"]
        reward_ko = data_dict["reward_ko"]
        reward_f = data_dict["reward_f"]
        # Update the derivatives performance plot
        derivates_fig = update_derivatives_performance_plot(
            data_dict,
            derivative_leverage,
            holding_period,
        )
        st.write(
            f"Win ratio over the past {constants.trading_days} trading days (KO / Factor):"
            + f" {win_ratio_ko:.1f}%/ {win_ratio_f:.1f}%. Average risk (loss)"
            + f" vs. average reward (win) per trade over the past {constants.trading_days}"
            + f" trading days (KO/ Factor): 1:{reward_ko:.2f}/ 1:{reward_f:.2f}"
        )
        st.plotly_chart(
            derivates_fig,
            use_container_width=True,
        )
        # Add a calibration plot for the derivatives performance
        calibration_fig = update_derivates_calibration_plot(
            data_dict, derivative_leverage, holding_period
        )
        st.plotly_chart(
            calibration_fig,
            use_container_width=True,
        )

    with tab3:
        # Header for the Stock Screener
        st.markdown("", unsafe_allow_html=True)
        st.write("## Stock Universe Screener (Work in Progress)")

        # TODO: Display the failed download log
        with st.spinner("Fetching data (might take up to 5 minutes)..."):
            data = download_universe()

        # Display the output in Streamlit
        st.success("Data fetched successfully!")
        # Filter data based on the "Volume" column
        filter_by_volume = st.checkbox(
            "Keep only Stocks with High Trading Volume", value=True
        )
        if filter_by_volume:
            # Filter data based on the "Volume" column
            sufficient_volume = (data["Volume"] > 100000).all(axis=0)
            # Remove columns with insufficient volume
            filtered_cols = [
                col
                for i, col in enumerate(data["Volume"].columns)
                if sufficient_volume.iloc[i] == True
            ]
        else:
            # Keep all columns
            filtered_cols = data["Volume"].columns

        # Display the past 60 days of the adjusted close data
        n_days_60 = 60
        adj_close_data = data.xs("Adj Close", level=0, axis=1).iloc[-(n_days_60 + 1) :]
        adj_close_data = adj_close_data[filtered_cols]
        n_assets = adj_close_data.shape[1]
        st.write(
            f"### Past {n_days_60} Days of Adjusted Close Data for {n_assets} Assets"
        )
        st.dataframe(data=adj_close_data.tail(10))

        # Header for the stock universe indicators
        st.write("## Indicators for the Stock Universe")
        st.markdown(
            """
            ### Kelly Criterion \n
            Filter for stocks with a low volatility compared to their past 60 day 
            (30 day) returns and display the average daily range (ADR) of the 
            corresponding stocks.
            """
        )

        # Slider for the risk free rate
        risk_free_rate_u = st.slider(
            "Risk Free Yearly Return [%] (Costs)",
            min_value=0.0,
            max_value=8.0,
            value=3.0,
            step=0.25,
            key="stock_universe",
        )

        # Calculate the Kelly leverage factor for the stock universe for the past n_days
        lev_60, lev_gt_10_60, largest_20_60 = kelly_selection(
            adj_close_data, risk_free_rate_u, n_days_60
        )
        # Repeat for a different time window
        n_days_30 = 30
        lev_30, lev_gt_10_30, largest_20_30 = kelly_selection(
            adj_close_data, risk_free_rate_u, n_days_30
        )

        # Calculate the average daily range for the stock universe
        high_data = data.xs("High", level=0, axis=1).iloc[-n_days_60:]
        high_data = high_data[filtered_cols]
        low_data = data.xs("Low", level=0, axis=1).iloc[-n_days_60:]
        low_data = low_data[filtered_cols]
        # Calculate the average daily range for the past n_days
        dr_60, adr_60 = compute_adr(high_data, low_data, n_days_60)
        dr_30, adr_30 = compute_adr(
            high_data.iloc[-n_days_30:], low_data.iloc[-n_days_30:], n_days_30
        )

        # Create columns for the indicators
        col1, col2, col3, col4 = st.columns(4)

        # Use the columns for display
        col1.write(
            f"Kelly leverage factor > 10 in the past {n_days_60} days:"
            f" {len(lev_gt_10_60)} Stocks. Top 20:"
        )
        col1.dataframe(data=largest_20_60)
        col2.write(
            f"Kelly leverage factor > 10 in the past {n_days_30} days:"
            f" {len(lev_gt_10_30)} Stocks. Top 20:"
        )
        col2.dataframe(data=largest_20_30)
        col3.write(f"Average daily range [%] in the past {n_days_60} days:")
        col3.dataframe(data=adr_60[largest_20_60.index])
        col4.write(f"Average daily range [%] in the past {n_days_30} days:")
        col4.dataframe(data=adr_30[largest_20_30.index])

        st.markdown(
            """
            ### Average Daily Range \n
            Filter for stocks with a high average daily range (ADR) in the past
            60 days (30 days) and a positive trend.
            """
        )

        # Filter for a positive trend in the past n_days
        ret_60, ret_10_60 = positive_return_selection(adj_close_data, n_days_60)
        top_adr_60 = adr_60[ret_10_60].nlargest(20).round(2)
        top_adr_60.name = "ADR [%]"
        ret_30, ret_10_30 = positive_return_selection(adj_close_data, n_days_30)
        top_adr_30 = adr_30[ret_10_30].nlargest(20).round(2)
        top_adr_30.name = "ADR [%]"

        # Create columns for the indicators
        col1, col2, col3, col4 = st.columns(4)

        # Use the columns for display
        col1.write(f"Top 20 ADR stocks in the past {n_days_60} days:")
        col1.dataframe(data=top_adr_60)
        col2.write(f"Top 20 ADR stocks in the past {n_days_30} days:")
        col2.dataframe(data=top_adr_30)
        col3.write(f"Return [%] past {n_days_60} days:")
        col3.dataframe(data=ret_60[top_adr_60.index])
        col4.write(f"Return [%] in the past {n_days_30} days:")
        col4.dataframe(data=ret_30[top_adr_30.index])

        st.markdown(
            """
            ### Custom Filters \n
            Filter stocks for custom Kelly leverage, average daily range (ADR), 
            and volatility in the past 60 days (30 days).
            """
        )
        # Slider for the Kelly filter
        kelly_filter = st.slider(
            "Min. Kelly Leverage [%]",
            min_value=2.5,
            max_value=30.0,
            value=5.0,
            step=2.5,
        )
        # Slider for the ADR filter
        adr_filter = st.slider(
            "Min. Average Daily Range (ADR) [%]",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
        )
        # Slider for the volatility filter
        volatility_filter = st.slider(
            "Max. Annualized Volatility [%]",
            min_value=5.0,
            max_value=100.0,
            value=15.0,
            step=5.0,
        )

        # Apply filters to the leverage factor
        filter_lev_60 = lev_60 > kelly_filter
        filter_lev_30 = lev_30 > kelly_filter
        # Apply filters to the average daily range
        filter_adr_60 = adr_60 > adr_filter
        filter_adr_30 = adr_30 > adr_filter
        # Apply filters to the volatility
        vol_60 = volatility_selection(adj_close_data, n_days_60)
        filter_vol_60 = vol_60 < volatility_filter
        vol_30 = volatility_selection(adj_close_data, n_days_30)
        filter_vol_30 = vol_30 < volatility_filter

        # Combine the filters
        combined_60 = filter_lev_60 & filter_adr_60 & filter_vol_60
        combined_30 = filter_lev_30 & filter_adr_30 & filter_vol_30

        # Create new dataframes for the filtered stocks
        filtered_data_60 = adj_close_data.iloc[-1].T.round(2).to_frame()[combined_60]
        filtered_data_60.columns = ["Adj. Close"]
        filtered_data_60["Adj. Close"] = filtered_data_60["Adj. Close"].astype(
            "float64"
        )
        filtered_data_60["Kelly Leverage"] = lev_60.round(2)[combined_60]
        filtered_data_60["ADR [%]"] = adr_60[combined_60]
        filtered_data_60["Volatility [%]"] = vol_60[combined_60]
        filtered_data_30 = adj_close_data.iloc[-1].T.round(2).to_frame()[combined_30]
        filtered_data_30.columns = ["Adj. Close"]
        filtered_data_30["Adj. Close"] = filtered_data_30["Adj. Close"].astype(
            "float64"
        )
        filtered_data_30["Kelly Leverage"] = lev_30.round(2)[combined_30]
        filtered_data_30["ADR [%]"] = adr_30[combined_30]
        filtered_data_30["Volatility [%]"] = vol_30[combined_30]

        # Sort the dataframes by the Kelly leverage
        filtered_data_60 = filtered_data_60.sort_values(
            by="Kelly Leverage", ascending=False
        )
        filtered_data_30 = filtered_data_30.sort_values(
            by="Kelly Leverage", ascending=False
        )

        # Create columns for the indicators
        col1, col2 = st.columns(2)

        # Use the columns for display
        col1.write(f"Filtered stocks ({n_days_60} day window): {len(filtered_data_60)}")
        col1.dataframe(data=filtered_data_60)
        col2.write(f"Filtered stocks ({n_days_30} day window): {len(filtered_data_30)}")
        col2.dataframe(data=filtered_data_30)
