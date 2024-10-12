import pandas as pd
import pytest

from vol_decay.utils.utils import (
    gmean,
    leveraged_return,
    performance_cumprod,
    validate_inputs,
)


def test_validate_inputs_regular_data():
    regular_data = [
        pd.Series([1, 2, 3, 4, 5]),
        pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
    ]
    for data in regular_data:
        validate_inputs(data)


def test_empty_data():
    empty_data = [
        pd.Series([]),
        pd.DataFrame({"A": [], "B": []}),
    ]
    for data in empty_data:
        with pytest.raises(
            ValueError, match="The input data contains missing values or is empty."
        ):
            validate_inputs(data)


def test_null_data():
    null_data = [
        pd.Series([1, 2, None, 4, 5]),
        pd.DataFrame({"A": [1, 2, 3], "B": [4, None, 6]}),
    ]
    for data in null_data:
        with pytest.raises(
            ValueError, match="The input data contains missing values or is empty."
        ):
            validate_inputs(data)


@pytest.mark.parametrize(
    "lev_factor, cagr_underlying, leverage_expense_ratio, libor, yearly_volatility, expected_result, expected_type",
    [
        (2.0, 0.07, 0.01, 0.02, 0.15, "positive", float),  # regular case
        (
            0.0,
            0.07,
            0.01,
            0.02,
            0.15,
            "positive",
            float,
        ),  # no investment but volatility
        (-2.0, 0.07, 0.01, 0.02, 0.15, "negative", float),  # negative leverage factor
        (2.0, 0.0, 0.01, 0.02, 0.15, "negative", float),  # no CAGR underlying
        (2.0, 0.07, 0.01, 0.02, 1.0, "negative", float),  # high volatility
        (2.0, 0.07, -0.01, 0.02, 0.15, "positive", float),  # negative expense ratio
        (2.0, 0.07, 0.01, -0.02, 0.15, "positive", float),  # negative LIBOR
    ],
)
def test_leveraged_return(
    lev_factor,
    cagr_underlying,
    leverage_expense_ratio,
    libor,
    yearly_volatility,
    expected_result,
    expected_type,
):
    result = leveraged_return(
        lev_factor, cagr_underlying, leverage_expense_ratio, libor, yearly_volatility
    )

    assert isinstance(result, expected_type)
    if expected_result == "positive":
        assert result > 0
    else:
        assert result > -1


@pytest.mark.parametrize(
    "x, time_window, days_in_year, expected",
    [
        (0.1, 1, 365, 0.00026115787606784124),  # Positive value
        (0.0, 1, 365, 0.0),  # Zero value
        (-0.1, 1, 365, -0.0002886172890222971),  # Negative value
        (0.1, 30, 365, 0.007864477220618893),  # Different time_window
        (0.1, 1, 360, 0.0002647855489630313),  # Different days_in_year
    ],
)
def test_gmean(x, time_window, days_in_year, expected):
    result = gmean(x, time_window, days_in_year)
    assert pytest.approx(result, rel=1e-9) == expected


@pytest.mark.parametrize(
    "returns, expected_result, expected_type",
    [
        (pd.Series([0.01, 0.02, 0.03]), 6.1106000000000105, float),  # positive returns
        (
            pd.Series([-0.01, -0.02, -0.03]),
            -5.8906000000000125,
            float,
        ),  # negative returns
        (
            pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [-0.01, -0.02, -0.03]}),
            pd.Series([6.1106000000000105, -5.8906000000000125], index=["A", "B"]),
            pd.Series,
        ),  # DataFrame
    ],
)
def test_cumulative_product(returns, expected_result, expected_type):
    result = performance_cumprod(returns)
    if isinstance(result, pd.Series):
        # ignore the name attribute
        expected_result.name = result.name
        pd.testing.assert_series_equal(result, expected_result)
    else:
        assert result == expected_result
    assert isinstance(result, expected_type)


def test_cumulative_product_empty_input():
    empty_data = [
        pd.Series([]),
        pd.DataFrame({"A": [], "B": []}),
    ]
    for data in empty_data:
        with pytest.raises(
            ValueError, match="The input data contains missing values or is empty."
        ):
            performance_cumprod(data)
