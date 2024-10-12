import pandas as pd
import pytest

from vol_decay.utils.kelly_calculations import kelly_crit


# calculate Kelly Criterion for regular (mixed) inputs
def test_kelly_crit_regular_inputs():
    yearly_er_list = [
        10.0,
        pd.Series([10.0, 8.0, 6.0]),
        pd.DataFrame(
            {
                "A": [10.0, 8.0, 6.0],
                "B": [9.0, 7.0, 5.0],
            }
        ),
    ]
    yearly_risk_free_list = [
        2.0,
        pd.Series([2.0, 1.5, 1.0]),
        pd.DataFrame(
            {
                "A": [2.0, 1.5, 1.0],
                "B": [1.5, 1.0, 0.5],
            }
        ),
    ]
    yearly_volatility_list = [
        5.0,
        pd.Series([5.0, 4.0, 3.0]),
        pd.DataFrame(
            {
                "A": [5.0, 4.0, 3.0],
                "B": [4.0, 3.0, 2.0],
            }
        ),
    ]
    for yearly_er in yearly_er_list:
        for yearly_risk_free in yearly_risk_free_list:
            for yearly_volatility in yearly_volatility_list:
                expected_kelly = (
                    100.0 * (yearly_er - yearly_risk_free) / (yearly_volatility**2)
                )
                result = kelly_crit(yearly_er, yearly_risk_free, yearly_volatility)

                if any(
                    isinstance(data, pd.DataFrame)
                    for data in (yearly_er, yearly_risk_free, yearly_volatility)
                ):
                    assert isinstance(result, pd.DataFrame)
                    pd.testing.assert_frame_equal(result, expected_kelly)
                elif any(
                    isinstance(data, pd.Series)
                    for data in (yearly_er, yearly_risk_free, yearly_volatility)
                ):
                    assert isinstance(result, pd.Series)
                    pd.testing.assert_series_equal(result, expected_kelly)
                else:
                    assert isinstance(result, float)
                    assert result == expected_kelly


# test zero volatility edge-case
def test_kelly_crit_zero_volatility():
    yearly_er = 10.0
    yearly_risk_free = 2.0
    yearly_volatility_list = [
        0.0,
        pd.Series([0.0, 0.0, 0.0]),
        pd.DataFrame(
            {
                "A": [0.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0],
            }
        ),
    ]
    for yearly_volatility in yearly_volatility_list:
        with pytest.raises(ValueError, match="Volatility must be positive."):
            kelly_crit(yearly_er, yearly_risk_free, yearly_volatility)


# test non-equal lengths edge-case
def test_kelly_crit_non_equal_lengths():
    yearly_er = pd.Series([10.0, 8.0, 6.0])
    yearly_risk_free = pd.Series([2.0, 1.5, 1.0])
    yearly_volatility = pd.Series([5.0, 4.0])
    with pytest.raises(ValueError, match="All inputs must have the same length"):
        kelly_crit(yearly_er, yearly_risk_free, yearly_volatility)
