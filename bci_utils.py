"""
BCI Utility Functions
=====================

Helper functions for Bellman Conformal Inference experiments,
including lambda_max estimation and parameter tuning.
"""

import numpy as np
import pandas as pd


def estimate_lambda_max_from_warmup(scores, warmup_periods=100, percentile=99):
    """
    Estimate lambda_max from warmup period (PI's suggestion for stability)
    
    Uses the percentile of conformity scores in the initial warmup period
    to set an upper bound for lambda, preventing instability.
    
    Args:
        scores: Array of conformity scores from baseline model
        warmup_periods: Number of initial periods to use (default: 100)
        percentile: Percentile for estimation (default: 99)
                   - 99: tight setting (more conservative)
                   - 95: loose setting (more aggressive)
    
    Returns:
        lambda_max: Estimated upper bound for lambda
    """
    if len(scores) < warmup_periods:
        print(f"Warning: Only {len(scores)} scores available, using all for warmup")
        warmup_periods = len(scores)
    
    warmup_scores = scores[:warmup_periods]
    lambda_max = np.percentile(warmup_scores, percentile)
    
    return lambda_max


def compute_metrics(result_df, task='rtfc', warmup=100):
    """
    Compute variance, miscoverage, std, and average length from experiment results
    
    Args:
        result_df: DataFrame with columns ['true_y', 'upper', 'lower', 'alpha', 'beta']
        task: Task type ('rtfc', 'vlfc', 'trend')
        warmup: Number of initial periods to skip (default: 100)
    
    Returns:
        variance: Variance of local miscoverage (for variance-matching tuning)
        miscoverage: Mean miscoverage rate (as fraction, not %)
        avg_length: Mean interval length
        frac_inf: Fraction of infinite intervals
        std_miscov: Standard deviation of LOCAL miscoverage (rolling window)
    """
    if result_df is None:
        return 1e10, 0.0, 0.0, 0.0, 0.0

    # Skip warmup period
    df = result_df.iloc[warmup:].copy()

    # Ensure numeric types
    alpha = pd.to_numeric(df['alpha'], errors='coerce')
    beta = pd.to_numeric(df['beta'], errors='coerce')
    upper = pd.to_numeric(df['upper'], errors='coerce')
    lower = pd.to_numeric(df['lower'], errors='coerce')

    # Coverage indicator for each time point (1 = miscovered, 0 = covered)
    err_ind = (alpha > beta).astype(float)
    
    # LOCAL MISCOVERAGE (rolling window average)
    local_miscov = err_ind.rolling(window=50, min_periods=1).mean()
    
    # VARIANCE (for tuning - variance of rolling local miscoverage)
    variance = local_miscov.var()
    
    # MEAN MISCOVERAGE (target should be ~10% for alpha=0.1)
    miscoverage = err_ind.mean()
    
    # STANDARD DEVIATION (for paper table)
    # *** FIX: std of ROLLING LOCAL MISCOVERAGE, not raw binary indicators ***
    std_miscov = local_miscov.std()  # This gives ~0.003 (0.3%), not 0.30 (30%)!
    
    # INTERVAL LENGTHS
    raw_lengths = upper - lower
    frac_inf = (~np.isfinite(raw_lengths)).mean()
    valid_lengths = raw_lengths[np.isfinite(raw_lengths)]
    avg_length = valid_lengths.mean() if len(valid_lengths) > 0 else 0.0

    return variance, miscoverage, avg_length, frac_inf, std_miscov

def compute_initial_miscov(result_df, window=100):
    """
    Compute initial miscoverage rate (for lambda_init tuning)
    
    Args:
        result_df: DataFrame with 'beta' column (coverage indicators)
        window: Number of initial periods to consider
    
    Returns:
        initial_miscov: Initial miscoverage rate (as fraction)
    """
    if result_df is None:
        return 1.0
    
    df = result_df.iloc[:window].copy()
    alpha = pd.to_numeric(df['alpha'], errors='coerce')
    beta = pd.to_numeric(df['beta'], errors='coerce')
    
    initial_miscov = (alpha > beta).astype(float).mean()
    return initial_miscov


def tune_lambda_init(bci_config, target_miscov, search_range=(0.001, 10.0), max_iters=20):
    """
    Tune BCI's lambda_init to match target initial miscoverage (e.g., from PID)
    
    Args:
        bci_config: BCI configuration dictionary
        target_miscov: Target initial miscoverage rate (from PID)
        search_range: (min, max) for binary search
        max_iters: Maximum binary search iterations
    
    Returns:
        optimal_lambda_init: Tuned lambda_init value
    """
    from experiment import ForecastingExperiment
    
    lo, hi = search_range
    best_lambda = lo
    best_diff = 1e10
    
    for _ in range(max_iters):
        mid = np.sqrt(lo * hi)  # Geometric mean for log-scale search
        
        # Test this lambda_init
        config_test = bci_config.copy()
        config_test['lambda_init'] = mid
        
        try:
            exp_test = ForecastingExperiment(config_test)
            exp_test.run()
            
            init_miscov = compute_initial_miscov(exp_test.result, window=100)
            diff = abs(init_miscov - target_miscov)
            
            if diff < best_diff:
                best_diff = diff
                best_lambda = mid
            
            # Update search bounds
            if init_miscov > target_miscov:
                hi = mid  # lambda_init too high, reduce
            else:
                lo = mid  # lambda_init too low, increase
        except:
            hi = mid  # On error, reduce lambda_init
    
    return best_lambda

def binary_search_bci_gamma(bci_config, target_var, dataset, task, tol=0.01, max_iter=20):
    """
    Binary search to find gamma that matches target variance
    
    Args:
        bci_config: Base BCI configuration dict
        target_var: Target variance to match (from ACI)
        dataset: Dataset name (for experiment config)
        task: Task name ('rtfc', 'trend', or 'vol')
        tol: Relative tolerance for convergence
        max_iter: Maximum search iterations
        
    Returns:
        best_gamma: Tuned gamma value
        best_diff: Final variance difference
    """
    from experiment import ForecastingExperiment
    
    lo, hi = 0.001, 10.0
    best_gamma, best_diff = lo, float('inf')
    
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        
        cfg = bci_config.copy()
        cfg['gamma'] = float(mid)
        
        exp = ForecastingExperiment(cfg)
        exp.run()
        
        var, _, _, _, _ = compute_metrics(exp.result, task)
        diff = abs(var - target_var)
        
        if diff < best_diff:
            best_diff = diff
            best_gamma = mid
        
        if diff < tol * target_var:
            break
        
        # Adjust search bounds
        if var > target_var:
            hi = mid
        else:
            lo = mid
    
    return best_gamma, best_diff