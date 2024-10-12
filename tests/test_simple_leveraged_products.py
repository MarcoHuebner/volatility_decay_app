import pandas as pd
import pytest

from vol_decay.utils.simple_leveraged_products import (
    SimplifiedFactor,
    SimplifiedKnockout,
)


def test_initialization_and_update_properties():
    for _class in [SimplifiedFactor, SimplifiedKnockout]:
        product = _class(expense_ratio=0.01, rel_transact_costs=0.02, hold_period=30)

        assert product.expense_ratio == 0.01
        assert product.rel_transact_costs == 0.02
        assert product.hold_period == 30
        assert product.percent == 100.0

        product.update_properties(expense_ratio=0.02, percent=90.0)
        assert product.expense_ratio == 0.02
        assert product.percent == 90.0


# test edge-case types
def test_invalid_returns_input_type():
    for _class in [SimplifiedFactor, SimplifiedKnockout]:
        product = _class(expense_ratio=0.01, rel_transact_costs=0.02, hold_period=30)

        with pytest.raises(
            TypeError, match="returns must be either a pandas Series or a numpy ndarray"
        ):
            product._apply_transaction_costs(returns="invalid_type")
        with pytest.raises(
            TypeError, match="returns must be either a pandas Series or a numpy ndarray"
        ):
            product._apply_transaction_costs(returns=pd.DataFrame())
