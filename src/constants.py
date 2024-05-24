"""
Define all the constants here.

"""

import numpy as np

# define the trading days in a year
trading_days = 252
five_years = 5 * trading_days

# define display functions
ann_return = 0.037 * trading_days
ann_risk_free = 0.005 * trading_days
ann_vol = 1.2 * np.sqrt(trading_days)
# define parameters (except for leverage all in percent)
lev_r = 2.0
exp_r = 0.6
libor = 0.5
