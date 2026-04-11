"""
Patches ForecastingExperiment.run() to handle T>1 and scalar comparisons
Import this after importing ForecastingExperiment to apply the patch
"""

import numpy as np
import pandas as pd


def to_scalar(x):
    """Convert array-like to scalar"""
    if isinstance(x, (list, np.ndarray, pd.Series)):
        return x[0]
    return x


def robust_run(self):
    """Patched run method for ForecastingExperiment"""
    self.result = pd.DataFrame(columns=['beta', 'alpha', 'upper', 'lower', 'lambda', 'index', 'true_y'])
    method = self.params['method']
    alpha0 = to_scalar(self.params['alpha0'])

    if method in ['aci', 'bci']:
        lambda_max = to_scalar(self.params.get('lambda_max', 1e6))
        lambda_min = to_scalar(self.params.get('lambda_min', 0))
        lbd        = to_scalar(self.params.get('lambda_init', 0.9))
        gamma      = to_scalar(self.params.get('gamma', 0.01))
        if method == 'bci':
            T_target = int(to_scalar(self.params.get('T', 1)))
            Tp       = int(to_scalar(self.params.get('Tp', 100)))

    self.fcdata.refresh()
    effective_T = None

    for t in range(self.fcdata.expectancy()):
        onedaydata = self.fcdata.next()
        if method == 'bci' and effective_T is None:
            K = len(onedaydata['nested_pred_sets'])
            effective_T = min(T_target, K)

        if method in ['aci', 'bci']:
            if t > 0:
                prev_alpha = to_scalar(self.result.loc[t-1]['alpha'])
                prev_beta  = to_scalar(self.result.loc[t-1]['beta'])
                lbd = lbd - gamma * (alpha0 - float(prev_alpha > prev_beta))

            if lbd >= lambda_max: 
                alpha = 0
            elif lbd <= lambda_min: 
                alpha = 1
            else:
                if method == 'bci':
                    if t > Tp:
                        history = self.result.loc[t - Tp:t - 1]
                        rhoTp = int(np.sum(history['alpha'].apply(to_scalar) > history['beta'].apply(to_scalar)))
                    else:
                        rhoTp = int(alpha0 * Tp)

                    from utils.dp import DynamicConformal
                    dc = DynamicConformal(
                        effective_T, alpha0,
                        [onedaydata['beta_cdf'] for _ in range(effective_T)],
                        [onedaydata['nested_pred_sets'][s].length() for s in range(effective_T)],
                        lbd, rhoTp, Tp
                    )
                    dc.dp(bins=200)
                    alpha = to_scalar(dc.optimal_policy[0].eval(0))
                else: 
                    alpha = 1 - lbd
        else: 
            alpha = alpha0

        new_row = {
            'beta':   to_scalar(onedaydata['beta']), 
            'alpha':  alpha,
            'upper':  to_scalar(onedaydata['nested_pred_sets'][0].upper.eval(alpha)),
            'lower':  to_scalar(onedaydata['nested_pred_sets'][0].lower.eval(alpha)),
            'lambda': lbd if method in ['aci', 'bci'] else None,
            'index':  onedaydata['index'], 
            'true_y': to_scalar(onedaydata['true_y']),
        }
        self.result = pd.concat([self.result, pd.DataFrame([new_row])], ignore_index=True)

    self.fcdata.refresh()
    self.result = self.result.set_index('index')


def apply_patch():
    """Apply the robust_run patch to ForecastingExperiment"""
    from experiment import ForecastingExperiment
    ForecastingExperiment.run = robust_run
    print("✓ ForecastingExperiment.run() patched")