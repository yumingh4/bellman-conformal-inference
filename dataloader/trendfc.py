"""
Google Trends Data Loader
Following the same pattern as normfc.py (ReturnData)
"""

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import sys
sys.path.append('.')

from utils.function01 import SymFunction01
from utils import make_nested_pred_sets_normal
from dataloader import ForecastingData

import pandas as pd
import numpy as np


class PandasNormalData(ForecastingData):
    def __init__(self, task, id, beta_cdf_len, value_name, forecasting_horizon):
        super().__init__(task, id)
    
        self.df = pd.read_csv('data/{}/{}-fc.csv'.format(task, id), index_col=0)
        self.df.index = pd.to_datetime(self.df.index, format='%m/%d/%Y')  # Match your CSV format

        # Data setup
        self.dates = self.df.index
        self.beta_cdf_len = beta_cdf_len
        self.curr_ind = self.beta_cdf_len
        self.end_ind = len(self.df)-1
        self.name = '{}-{}'.format(task, id)
        self.value_name = value_name
        self.horizon = forecasting_horizon

    def next(self):
        if self.curr_ind > self.end_ind:
            return False
        else:
            date = self.dates[self.curr_ind]
            true_y = float(self.df.loc[date][self.value_name])

            # forecasting data            
            preds_str = ['pred_{}'.format(k) for k in range(1, self.horizon+1)]
            ses_str = ['se_{}'.format(k) for k in range(1, self.horizon+1)]
            mus = np.array(self.df.loc[date][preds_str])
            sigmas = np.array(self.df.loc[date][ses_str])

            # make nps
            nps = make_nested_pred_sets_normal(sigmas, mus, len(ses_str))

            # make empirical beta dist
            start_date = self.dates[self.curr_ind-self.beta_cdf_len+1]
            betas = np.array(self.df[start_date:date][['beta']]).reshape(-1)
            beta_cdf = SymFunction01('np.mean(betas.reshape(-1, 1)<alpha, axis=0)',
                                     'alpha', {'betas': betas})

            # update curr_ind
            self.curr_ind = self.curr_ind + 1
            return {'true_y': true_y,
                    'beta': self.df.loc[date]['beta'],
                    'nested_pred_sets': nps,
                    'beta_cdf': beta_cdf,
                    'index': date}

    def refresh(self):
        self.curr_ind = self.beta_cdf_len

    def expectancy(self):
        return self.end_ind - self.curr_ind


class TrendData(PandasNormalData):
    """
    Google Trends data loader
    Uses 7-step ahead forecasting horizon (pred_1 through pred_7)
    """
    def __init__(self, topic_id, beta_cdf_len=100):
        super().__init__('trend', topic_id, beta_cdf_len, 'log_value', 7)


if __name__ == '__main__':
    # Test the dataloader
    exp = TrendData('statistics')  # or 'deeplearning', 'iphone', etc.
    exp.plot_ecc()