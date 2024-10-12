"""
Define simplified leveraged products.

"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from vol_decay.utils.utils import gmean, validate_inputs


class SimpleLeveragedProducts(ABC):
    def __init__(
        self,
        expense_ratio: float,
        rel_transact_costs: float,
        hold_period: int,
        percent: float = 100.0,
    ):
        self.expense_ratio = expense_ratio
        self.rel_transact_costs = rel_transact_costs
        self.hold_period = hold_period
        self.percent = percent

    def _apply_expense_ratio(
        self, returns: pd.Series | np.ndarray
    ) -> pd.Series | np.ndarray:
        return returns + gmean(-self.expense_ratio / self.percent)

    def _apply_transaction_costs(
        self, returns: pd.Series | np.ndarray
    ) -> pd.Series | np.ndarray:
        # determine the type of returns and set the appropriate indexing method
        if isinstance(returns, pd.Series):
            get_item = returns.iloc
        elif isinstance(returns, np.ndarray):
            get_item = returns
        else:
            raise TypeError("returns must be either a pandas Series or a numpy ndarray")

        # Apply transaction costs
        get_item[0] -= self.rel_transact_costs / self.percent
        if len(returns) >= self.hold_period:
            get_item[-1] -= self.rel_transact_costs / self.percent

        return returns

    def update_properties(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_daily_returns() -> pd.Series:
        pass


class SimplifiedFactor(SimpleLeveragedProducts):
    def __init__(
        self,
        expense_ratio: float,
        rel_transact_costs: float,
        hold_period: int,
        percent: float = 100.0,
    ):
        super().__init__(expense_ratio, rel_transact_costs, hold_period, percent)

    def get_daily_returns(
        self,
        daily_returns: pd.Series,
        daily_low_returns: pd.Series,
        expense_ratio: float | None = None,
        rel_transact_costs: float | None = None,
        hold_period: int | None = None,
        leverage: float = 1.0,
    ) -> pd.Series:
        """
        Calculate the daily returns of a factor with leverage using a simplified model.

        :param daily_returns: pd.Series, daily returns of the underlying asset
        :param daily_low_returns: pd.Series, daily low returns of the underlying asset
        :param expense_ratio: float, expense ratio of the factor (in percent)
        :param rel_transact_costs: float, relative transaction costs of the factor
        :param hold_period: int, number of days the factor is held
        :param leverage: float, leverage of the factor
        :return: pd.Series, daily returns of the factor with leverage
        """
        if not isinstance(leverage, float):
            raise TypeError("The leverage must be a float.")
        validate_inputs(daily_returns)
        validate_inputs(daily_low_returns)

        if expense_ratio is not None:
            self.update_properties(expense_ratio=expense_ratio)
        if rel_transact_costs is not None:
            self.update_properties(rel_transact_costs=rel_transact_costs)
        if hold_period is not None:
            self.update_properties(hold_period=hold_period)

        # compute daily returns of the factor product
        leveraged_daily_returns = self._apply_expense_ratio(daily_returns * leverage)

        # get first intra-day knockout event (if it exists)
        # add 5% safety margin for the issuer, when the knockout is triggered
        cutoff_margin = 0.05
        mask = (daily_low_returns.values * leverage) <= (-1 + cutoff_margin)

        if mask.any():
            # set all following returns to negative one to obtain a zero cumprod
            # if idxmax is a "true" value (i.e. the first knockout event)
            index = np.argmax(mask)
            leveraged_daily_returns.iloc[index:] = -1
        else:
            leveraged_daily_returns = self._apply_transaction_costs(
                leveraged_daily_returns
            )

        return leveraged_daily_returns


class SimplifiedKnockout(SimpleLeveragedProducts):
    def __init__(
        self,
        expense_ratio: float,
        rel_transact_costs: float,
        hold_period: int,
        percent: float = 100.0,
    ):
        super().__init__(expense_ratio, rel_transact_costs, hold_period, percent)

    def get_daily_returns(
        self,
        price: pd.Series,
        low_price: pd.Series,
        expense_ratio: float | None = None,
        rel_transact_costs: float | None = None,
        hold_period: int | None = None,
        initial_leverage: float = 1.0,
    ) -> pd.Series:
        """
        Calculate the daily returns of a knockout product using a simplified model.
        Working with closing prices, this supposes the knockout was bought at the
        closing course of the first day, making zero returns on the first day.

        :param price: pd.Series, price of the underlying asset
        :param low_price: pd.Series, low price of the underlying asset
        :param expense_ratio: float, expense ratio of the knockout product (in percent)
        :param rel_transact_costs: float, relative transaction costs of the knockout product
        :param hold_period: int, number of days the knockout product is held
        :param initial_leverage: float, initial leverage factor of the knockout product
        :return: pd.Series, daily returns of the knockout product
        """
        if not isinstance(initial_leverage, float):
            raise TypeError("The leverage must be a float.")
        validate_inputs(price)
        validate_inputs(low_price)

        if expense_ratio is not None:
            self.update_properties(expense_ratio=expense_ratio)
        if rel_transact_costs is not None:
            self.update_properties(rel_transact_costs=rel_transact_costs)
        if hold_period is not None:
            self.update_properties(hold_period=hold_period)

        # compute knockout barrier, incl. expense ratio estimation (all contained in buy)
        price_np = price.values
        ko_val = price_np[0] * (1 - (1 / initial_leverage))

        # compute daily returns
        pct_change = np.diff(price_np - ko_val) / (price_np[:-1] - ko_val) + gmean(
            -self.expense_ratio / self.percent
        )

        # get first knockout event (if it exists) - include intra-day knockouts
        mask = (price_np <= ko_val) | (
            np.concatenate(([price_np[0]], low_price.values[1:])) <= ko_val
        )

        if mask.any():
            # set all following returns to negative one to obtain a zero cumprod
            # if idxmax is a "true" value (i.e. the first knockout event)
            index = np.argmax(mask)
            pct_change[index:] = -1
        else:
            pct_change = self._apply_transaction_costs(pct_change)

        return pd.Series(pct_change, index=price.index[1:])
