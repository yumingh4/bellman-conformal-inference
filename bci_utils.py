"""
Shared utilities for BCI experiments
Extracts common functions from demo copy.ipynb
"""

import numpy as np
import pandas as pd
from experiment import ForecastingExperiment
from pid_external.pid_methods import quantile_integrator_log


def to_scalar(x):
    """Convert array-like to scalar"""
    if isinstance(x, (list, np.ndarray, pd.Series)):
        return x[0]
    return x


def compute_metrics(result_df, task):
    """Compute variance, miscoverage, length for a result dataframe"""
    alpha = pd.to_numeric(result_df['alpha'], errors='coerce')
    beta  = pd.to_numeric(result_df['beta'],  errors='coerce')
    upper = pd.to_numeric(result_df['upper'], errors='coerce')
    lower = pd.to_numeric(result_df['lower'], errors='coerce')
    
    err_ind      = (alpha > beta).astype(float)
    local_miscov = err_ind.rolling(window=50, min_periods=1).mean()
    miscov       = err_ind.mean()
    std          = local_miscov.std()
    
    if task == 'vlfc':
        raw_lengths = np.sqrt(np.maximum(upper, 0)) - np.sqrt(np.maximum(lower, 0))
    else:
        raw_lengths = upper - lower
        
    valid_lengths = raw_lengths[np.isfinite(raw_lengths)]
    length       = valid_lengths.mean() if len(valid_lengths) > 0 else np.nan
    
    return miscov, std, length


def compute_initial_miscov(result_df, window=100):
    """Compute average miscoverage over first window timesteps"""
    df    = result_df.iloc[:window].copy()
    alpha = pd.to_numeric(df['alpha'], errors='coerce')
    beta  = pd.to_numeric(df['beta'],  errors='coerce')
    return (alpha > beta).astype(float).mean()


def tune_lambda_init(bci_config, target_miscov, tol=1e-4, max_iter=30):
    """
    Binary-search lambda_init so BCI's initial miscoverage matches target_miscov.
    Higher lambda → wider intervals → lower initial miscoverage.
    """
    from experiment import ForecastingExperiment
    
    lo, hi    = 1e-4, 1e4
    best_lam  = bci_config.get('lambda_init', 0.9)
    best_diff = 1e10

    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        cfg = dict(bci_config)
        cfg['lambda_init'] = mid

        try:
            exp = ForecastingExperiment(cfg)
            exp.run()
            init_miscov = compute_initial_miscov(exp.result)
            diff = abs(init_miscov - target_miscov)
            
            if diff < best_diff:
                best_diff = diff
                best_lam  = mid
            if diff < tol:
                break
                
            if init_miscov > target_miscov:
                lo = mid   # increase lambda to widen intervals
            else:
                hi = mid   # decrease lambda to narrow intervals
        except Exception:
            hi = mid

    return best_lam


def generate_pid_baseline(exp_fixed, task):
    """Generate PID baseline with standard parameters"""
    y_true = pd.to_numeric(exp_fixed.result['true_y'], errors='coerce').values
    u_f = pd.to_numeric(exp_fixed.result['upper'], errors='coerce').values
    l_f = pd.to_numeric(exp_fixed.result['lower'], errors='coerce').values

    if task == 'vlfc':
        yt = np.sqrt(np.maximum(y_true, 0))
        ut = np.sqrt(np.maximum(u_f, 0))
        lt = np.sqrt(np.maximum(l_f, 0))
    else:
        yt, ut, lt = y_true, u_f, l_f

    y_pred_center = (ut + lt) / 2
    scores = np.abs(yt - y_pred_center)

    T_len = len(scores)
    ki = np.percentile(scores, 99)
    csat = max((2 / np.pi) * (np.ceil(np.log(T_len) * 0.05) - 1 / np.log(T_len)), 0.1)

    pid_out = quantile_integrator_log(
        scores, alpha=0.1, lr=0.0799, Csat=csat, KI=ki, ahead=1, T_burnin=100
    )
    q = np.maximum(np.array(pid_out['q']), 0.0)

    df_pid = exp_fixed.result.copy()
    if task == 'vlfc':
        df_pid['upper'] = (y_pred_center + q)**2
        df_pid['lower'] = np.maximum(y_pred_center - q, 0)**2
    else:
        df_pid['upper'] = y_pred_center + q
        df_pid['lower'] = y_pred_center - q

    df_pid['alpha'] = 0.1
    df_pid['beta'] = ((pd.to_numeric(df_pid['true_y'], errors='coerce').values >= df_pid['lower']) & 
                      (pd.to_numeric(df_pid['true_y'], errors='coerce').values <= df_pid['upper'])).astype(int)
    return df_pid