import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import sys
sys.path.append('.') 

from utils import make_nps_chi2
from utils.function01 import SymFunction01
from dataloader import ForecastingData

import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm

COMPANIES = ['AMD', 'Amazon', 'Nvidia']

def preprocess(cid):
    """
    Reading the raw stock data with robust error handling
    """
    try:
        df = pd.read_csv('data/raw/{}.csv'.format(cid))
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw data file not found for {cid}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty data file for {cid}")
    except Exception as e:
        raise RuntimeError(f"Error reading data for {cid}: {str(e)}")
    
    # Validate required columns
    required_cols = ['Date', ' Open']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} in {cid} data")
    
    # Convert and validate dates
    try:
        df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    except ValueError:
        raise ValueError(f"Invalid date format in {cid} data")
    
    df = df.sort_values(by='Date')
    
    # Remove duplicate dates
    if df['Date'].duplicated().any():
        logging.warning(f"Duplicate dates found in {cid} data, keeping last occurrence")
        df = df.drop_duplicates(subset=['Date'], keep='last')

    # Handle missing values
    if df[' Open'].isna().any():
        logging.warning(f"Missing Open prices in {cid} data, forward filling")
        df[' Open'] = df[' Open'].fillna(method='ffill')
        # If still NaN at beginning, backward fill
        df[' Open'] = df[' Open'].fillna(method='bfill')
    
    # Remove rows with zero prices (to avoid division by zero)
    zero_prices = (df[' Open'] == 0).any()
    if zero_prices:
        logging.warning(f"Zero prices found in {cid} data, removing affected rows")
        df = df[df[' Open'] != 0]
    
    # Ensure minimum data length
    if len(df) < 110:  # Need at least 100 + some buffer
        raise ValueError(f"Insufficient data length for {cid}: {len(df)} rows")

    # Volatility computation
    df.loc[:,'Pt'] = df[' Open']
    Ptp1 = np.array(df[' Open'][1:])
    df = df[:-1].copy()
    df.loc[:,'Ptp1'] = Ptp1
    df.loc[:,'1e2Rt'] = 100*(df['Ptp1'] - df['Pt'])/df['Pt']
    df.loc[:,'1e2Vt'] = np.abs(df['1e2Rt'])

    # Handle any remaining NaN/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    if df[['1e2Rt', '1e2Vt']].isna().any().any():
        logging.warning(f"NaN values in computed returns for {cid}, removing affected rows")
        df = df.dropna(subset=['1e2Rt', '1e2Vt'])
    
    # Validate final data
    if len(df) < 100:
        raise ValueError(f"Insufficient valid data after cleaning for {cid}: {len(df)} rows")

    df.loc[:,'1e4Vt2'] = df['1e2Vt']**2
    df = df.set_index('Date')
    df = df[['1e2Vt', '1e4Vt2', '1e2Rt']]

    return df


def forecast(returns, T, start_date, curr_date):
    """
    Use the return information returns[start_date, curr_date] to estimate the
    (a) GARCH(1, 1) parameter on sigma^2 for
        curr_date+1, curr_date+2, ..., curr_date+T .
    (b) the estimate of mu
    """
    train = returns[start_date:curr_date][['1e2Rt']]
    
    # Validate training data
    if len(train) < 50:  # Minimum for GARCH estimation
        raise ValueError(f"Insufficient training data: {len(train)} observations")
    
    if train.isna().any().any():
        logging.warning("NaN values in training data, removing")
        train = train.dropna()
        if len(train) < 50:
            raise ValueError(f"Insufficient valid training data after NaN removal: {len(train)}")
    
    # Check for constant returns (GARCH fails with zero variance)
    if train['1e2Rt'].var() == 0:
        logging.warning("Constant returns detected, using fallback method")
        # Fallback: simple historical volatility
        fallback_vol = 0.01  # Default small volatility
        sigma2Kpt = np.full(T, fallback_vol**2)
        muhat = train['1e2Rt'].mean()
        return sigma2Kpt, muhat
    
    try:
        garch11 = arch_model(train, p=1, o=0, q=1, dist='normal')
        res = garch11.fit(update_freq=100, disp='off')
        
        # Check for convergence issues
        if not res.converged:
            logging.warning("GARCH model did not converge, using fallback")
            fallback_vol = train['1e2Rt'].std()
            sigma2Kpt = np.full(T, fallback_vol**2)
            muhat = train['1e2Rt'].mean()
            return sigma2Kpt, muhat
        
        sigma2Kpt = res.forecast(horizon=T,
                                 start=train.last_valid_index(),
                                 reindex=False).variance
        sigma2Kpt = np.array(sigma2Kpt).reshape(-1)
        
        # Validate forecasts
        if np.any(sigma2Kpt <= 0) or np.any(np.isnan(sigma2Kpt)) or np.any(np.isinf(sigma2Kpt)):
            logging.warning("Invalid GARCH forecasts, using fallback")
            fallback_vol = train['1e2Rt'].std()
            sigma2Kpt = np.full(T, fallback_vol**2)
        
        muhat = res.params['mu']
        
        # Validate muhat
        if np.isnan(muhat) or np.isinf(muhat):
            muhat = train['1e2Rt'].mean()
            
    except Exception as e:
        logging.warning(f"GARCH fitting failed: {str(e)}, using fallback method")
        fallback_vol = train['1e2Rt'].std()
        sigma2Kpt = np.full(T, fallback_vol**2)
        muhat = train['1e2Rt'].mean()
    
    return sigma2Kpt, muhat


def make_forecasted_data(cid):
    df = preprocess(cid)
    dates = df.index

    m = 100 # length of fitting window: amount of past data to use
    horizon = 14 # length of forecasting into the future

    prev_nps = None
    for i in tqdm(range(m, len(df))):
        start_date = dates[i-m]
        end_date = dates[i-1]
        date = dates[i]
        sigma2hatKpt, muhat = forecast(df, horizon, start_date, end_date)
        df.loc[date, 'muhat'] = muhat
        for j in range(1, horizon+1):
            df.loc[date, 'sigma2_{}'.format(j)] = sigma2hatKpt[j-1]
        nps = make_nps_chi2(sigma2hatKpt, muhat)
        
        if prev_nps:
            true_Ytp1 = float(df.loc[end_date]['1e4Vt2'])
            beta = prev_nps[0].beta_threshold(true_Ytp1)
            df.loc[end_date, 'beta'] = beta
        prev_nps = nps
        
    df = df.dropna()
    df.to_csv('data/vlfc/{}-fc.csv'.format(cid))


class VolatilityData(ForecastingData):
    def __init__(self, cid, beta_cdf_len=100):
        super().__init__('vlfc', cid)
        
        # load forecasting df for the experiemnt
        df = pd.read_csv('data/vlfc/{}-fc.csv'.format(cid))
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d') # yuming changed this 
        df.sort_values(by='Date')
        self.df = df.set_index('Date')

        # experiment setup
        self.dates = self.df.index
        self.beta_cdf_len = beta_cdf_len
        self.curr_ind = self.beta_cdf_len
        self.end_ind = len(self.df)-1
        self.name = 'vlfc-{}'.format(cid)

    def next(self):
        if self.curr_ind > self.end_ind:
            return False
        else:
            try:
                date = self.dates[self.curr_ind]
                
                # Validate data availability
                if date not in self.df.index:
                    logging.warning(f"Date {date} not found in data, skipping")
                    self.curr_ind += 1
                    return self.next()  # Recursive call to skip
                
                row_data = self.df.loc[date]
                if pd.isna(row_data['1e4Vt2']):
                    logging.warning(f"Missing volatility data for {date}, skipping")
                    self.curr_ind += 1
                    return self.next()
                
                true_Vt2 = float(row_data['1e4Vt2'])
                
                # Validate true value
                if np.isnan(true_Vt2) or np.isinf(true_Vt2):
                    logging.warning(f"Invalid volatility value for {date}, skipping")
                    self.curr_ind += 1
                    return self.next()
    
                # forecasting data
                muhat = row_data['muhat']
                if np.isnan(muhat) or np.isinf(muhat):
                    logging.warning(f"Invalid muhat for {date}, using fallback")
                    muhat = 0.0
                
                sigma_columns = [col for col in 
                               self.df.columns if col.startswith('sigma2_')]
                sigma_columns.sort(key=lambda x: int(x.split('_')[1]))
                sigma2hatKpt = np.array(row_data[sigma_columns].values)
                
                # Validate sigma forecasts
                if np.any(sigma2hatKpt <= 0) or np.any(np.isnan(sigma2hatKpt)) or np.any(np.isinf(sigma2hatKpt)):
                    logging.warning(f"Invalid sigma forecasts for {date}, using fallback")
                    sigma2hatKpt = np.full(len(sigma_columns), 0.01)  # Default small variance
    
                # make nps
                nps = make_nps_chi2(sigma2hatKpt, muhat)
    
                # make empirical beta dist
                start_date = self.dates[self.curr_ind- self.beta_cdf_len]
                end_date = self.dates[self.curr_ind- 1]
                
                # Validate beta data availability
                beta_data = self.df[start_date:end_date][['beta']]
                if beta_data.isna().any().any():
                    logging.warning(f"Missing beta data for {date}, using fallback")
                    betas = np.full(self.beta_cdf_len, 0.5)  # Default beta
                else:
                    betas = np.array(beta_data).reshape(-1)
                
                beta_cdf = SymFunction01('np.mean(betas.reshape(-1, 1)<alpha, axis=0)',
                                         'alpha', {'betas': betas})
    
                # Validate beta value
                beta_val = row_data['beta']
                if np.isnan(beta_val) or np.isinf(beta_val):
                    beta_val = 0.5  # Default fallback
    
                # update curr_ind
                self.curr_ind += 1
                return {'true_y': true_Vt2,
                        'beta': beta_val,
                        'nested_pred_sets': nps,
                        'beta_cdf': beta_cdf,
                        'index': date}
            
            except Exception as e:
                logging.error(f"Error processing date {date}: {str(e)}")
                self.curr_ind += 1
                return self.next()  # Skip problematic date

    def refresh(self):
        self.curr_ind = self.beta_cdf_len

    def expectancy(self):
        return self.end_ind - self.curr_ind


if __name__ == '__main__':
    # make_forecasted_data('Amazon')
    # for cid in COMPANIES:
    # make_forecasted_data(cid)
    exp = VolatilityData('Amazon')
    exp.plot_ecc()