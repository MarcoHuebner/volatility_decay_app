from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from vol_decay.investments import InvestmentData


@pytest.fixture
def mock_data():
    return {
        "price": pd.Series(np.random.randn(100)),
        "low": pd.Series(np.random.randn(100)),
        "earnings": {"Reported EPS": pd.Series(np.random.randn(10))},
    }


@pytest.fixture
def investment_data():
    return InvestmentData("AAPL", 2.0)


def test_initialization(investment_data):
    assert investment_data.ticker == "AAPL"
    assert investment_data.risk_free_rate_for_ticker == 2.0

    print(investment_data.result_dict.keys())

    # check if result_dict contains the expected keys
    expected_keys = [
        "name",
        "price",
        "low",
        "ann_volatility",
        "30_d_volatility_vix",
        "ma50",
        "ma200",
        "earnings",
    ]
    assert set(investment_data.result_dict.keys()) == set(expected_keys)

    # check if the values in result_dict have the correct data types
    assert isinstance(investment_data.result_dict["name"], str)
    assert isinstance(investment_data.result_dict["price"], pd.Series)
    assert isinstance(investment_data.result_dict["low"], pd.Series)
    assert isinstance(investment_data.result_dict["ann_volatility"], pd.Series)
    assert isinstance(investment_data.result_dict["30_d_volatility_vix"], pd.Series)
    assert isinstance(investment_data.result_dict["ma50"], pd.Series)
    assert isinstance(investment_data.result_dict["ma200"], pd.Series)
    assert isinstance(investment_data.result_dict["earnings"], pd.DataFrame)

    assert not investment_data.pct_change.empty
    assert not investment_data.pct_change_low.empty


@patch("vol_decay.investments.get_prophet_forecast")
def test_get_forecast(mock_get_prophet_forecast, investment_data):
    mock_get_prophet_forecast.return_value = ("forecast_current", "forecast_old")
    forecast = investment_data.get_forecast()
    assert forecast == ("forecast_current", "forecast_old")
    mock_get_prophet_forecast.assert_called_once_with(investment_data.price)


@patch("vol_decay.investments.kelly_leverage")
def test_get_kelly_leverage(mock_kelly_leverage, investment_data):
    mock_kelly_leverage.return_value = pd.Series(np.random.randn(100))
    leverage = investment_data.get_kelly_leverage(30)
    assert leverage.equals(mock_kelly_leverage.return_value.tail(100))
    mock_kelly_leverage.assert_called_once_with(
        investment_data.pct_change, 2.0, time_window=30
    )


def test_get_par(investment_data):
    par_5, par_1 = investment_data.get_par()
    assert isinstance(par_5, float)
    assert isinstance(par_1, float)
    assert par_5 >= par_1


# test invalid ticker handling
@patch("vol_decay.utils.data_and_forecast.fetch_ticker_data")
def test_handle_invalid_ticker(mock_get_ticker_data):
    symbol = "INVALID"
    error_str = f"Ticker {symbol} not found. Please check the ticker symbol."
    mock_get_ticker_data.side_effect = ValueError(error_str)
    with pytest.raises(ValueError, match=error_str):
        InvestmentData(ticker="INVALID", risk_free_rate_for_ticker=2.0)
