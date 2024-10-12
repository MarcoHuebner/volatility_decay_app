import numpy as np
import pandas as pd
import pytest

from vol_decay import constants
from vol_decay.utils.volatility_calculations import empirical_annualized_volatility


@pytest.fixture
def multi_year_series():
    np.random.seed(0)
    return pd.Series(
        np.random.normal(loc=100, scale=5, size=constants.trading_days * 5)
    )


@pytest.fixture
def multi_year_dataframe():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "A": np.random.normal(loc=100, scale=5, size=constants.trading_days * 5),
            "B": np.random.normal(loc=200, scale=10, size=constants.trading_days * 5),
        }
    )


# calculate annualized volatility for a complete dataset without missing values
def test_series(multi_year_series):
    result = empirical_annualized_volatility(multi_year_series)
    assert isinstance(result, pd.Series)
    assert not result.empty
    expected = (
        multi_year_series.pct_change()
        .rolling(window=constants.trading_days)
        .std()
        .dropna()
        * np.sqrt(constants.trading_days)
        * 100
    )
    pd.testing.assert_series_equal(result, expected)


def test_dataframe(multi_year_dataframe):
    result = empirical_annualized_volatility(multi_year_dataframe)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected = (
        multi_year_dataframe.pct_change()
        .rolling(window=constants.trading_days)
        .std()
        .dropna()
        * np.sqrt(constants.trading_days)
        * 100
    )
    pd.testing.assert_frame_equal(result, expected)


def test_different_window():
    data = pd.Series([100, 101, 102, 103, 104, 105])
    window = 3
    result = empirical_annualized_volatility(data, window=window)
    assert isinstance(result, pd.Series)
    assert not result.empty
    expected = (
        data.pct_change().rolling(window=window).std().dropna()
        * np.sqrt(constants.trading_days)
        * 100
    )
    pd.testing.assert_series_equal(result, expected)


# test edge-cases
def test_empty_data():
    data = pd.Series([])
    # Assert that a ValueError is raised due to empty data
    with pytest.raises(ValueError):
        empirical_annualized_volatility(data)


def test_single_data_point():
    data = pd.Series([100])
    result = empirical_annualized_volatility(data)
    expected = pd.Series([], dtype=np.float64)
    pd.testing.assert_series_equal(result, expected)


def test_missing_values():
    data = pd.Series([100, None, 101, 105, None, 110])
    # Assert that a ValueError is raised due to missing values
    with pytest.raises(ValueError):
        empirical_annualized_volatility(data)
