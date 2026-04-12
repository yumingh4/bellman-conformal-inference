"""
Returns Forecasting - Calibrated Conformal Prediction Experiments
===================================================================
"""

# ============================================================================
# FORCE RELOAD OF BCI_UTILS (CRITICAL - MUST BE FIRST!)
# ============================================================================
import sys
if 'bci_utils' in sys.modules:
    del sys.modules['bci_utils']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_yaml
from experiment import ForecastingExperiment

# Force reload bci_utils
import importlib
import bci_utils
importlib.reload(bci_utils)

from bci_utils import (
    compute_metrics,
    compute_initial_miscov, 
    tune_lambda_init,
    estimate_lambda_max_from_warmup,
    binary_search_bci_gamma  # *** ADDED: Missing import ***
)

from visualize import gen_plot_data
from pid_external.pid_methods import quantile_integrator_log


# ============================================================================
# VERIFY COMPUTE_METRICS IS CORRECT
# ============================================================================
print("\n" + "="*70)
print("TESTING compute_metrics function...")
print("="*70)

test_df = pd.DataFrame({
    'alpha': [0.1] * 200,
    'beta': [1, 0] * 100,  # Alternating: 50% miscoverage
    'upper': [1.0] * 200,
    'lower': [0.0] * 200,
    'true_y': [0.5] * 200
})

result = compute_metrics(test_df, 'rtfc')
print(f"✓ Returns {len(result)} values")

if len(result) == 5:
    var, miscov, length, frac_inf, std = result
    print(f"✓ Unpacked: var={var:.4f}, miscov={miscov:.2%}, len={length:.4f}, frac_inf={frac_inf:.2%}, std={std:.4f}")
    
    # NEW TEST: std should be SMALL (rolling window std, not raw binary std)
    # For alternating 0/1 pattern with window=50, expect std ~ 0.05
    if std < 0.1:  # Should be much less than raw binary std (~0.50)
        print(f"✓✓ STD IS CORRECT! (rolling window std={std:.4f}, as expected)")
        print(f"   (This is std of LOCAL miscoverage, not raw binary indicators)")
    else:
        print(f"✗✗ STD IS WRONG! (std={std:.4f}, should be < 0.1 for rolling window)")
        print("ERROR: Still using raw binary std instead of rolling window std!")
        sys.exit(1)
else:
    print(f"✗✗ ERROR: Returns {len(result)} values instead of 5!")
    sys.exit(1)

print("="*70)
print("✓ compute_metrics verified - continuing with experiments...\n")

# Global constants aligned with paper
ALPHA = 0.1              # Target miscoverage = 10%
WARMUP_PERIODS = 100     # For lambda_max estimation
PERCENTILE = 99          # For lambda_max = 99th percentile of warmup scores



def run_returns_experiments(save_plots=True, verbose=True, results_dir='results/returns'):
    """
    Run calibrated returns forecasting experiments for MAIN BODY results
    
    Args:
        save_plots: Whether to save publication-quality figures
        verbose: Print detailed progress
        results_dir: Directory to save plots and tables
        
    Returns:
        all_rows: List of dicts with detailed metrics for each dataset/gamma
    """
    
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    datasets = ['Amazon', 'AMD', 'Nvidia']
    task = 'rtfc'
    
    # Testing different gamma values (will be normalized to c later)
    gammas_raw = [0.1, 0.008]
    
    all_rows = []

    for dataset in datasets:
        if verbose:
            print(f"\n{'='*70}")
            print(f"{dataset.upper()} — RETURNS FORECASTING")
            print(f"{'='*70}")

        # ================================================================
        # STEP 1: Run Fixed Baseline (for conformity scores)
        # ================================================================
        exp_fixed = ForecastingExperiment(
            read_yaml(f'config/{task}-fixed-{dataset}.yaml')
        )
        exp_fixed.run()

        # Extract conformity scores
        y_true = pd.to_numeric(exp_fixed.result['true_y'], errors='coerce').values
        upper = pd.to_numeric(exp_fixed.result['upper'], errors='coerce').values
        lower = pd.to_numeric(exp_fixed.result['lower'], errors='coerce').values
        y_pred_center = (upper + lower) / 2
        scores = np.abs(y_true - y_pred_center)

        # ================================================================
        # STEP 2: Estimate lambda_max from warmup (PI's suggestion)
        # ================================================================
        lambda_max = estimate_lambda_max_from_warmup(
            scores, 
            warmup_periods=WARMUP_PERIODS,
            percentile=PERCENTILE
        )
        
        if verbose:
            print(f"\n  λ_max (99th %ile, warmup={WARMUP_PERIODS}): {lambda_max:.4f}")

        # PID hyperparameters (dataset-specific)
        T_len = len(scores)
        csat = max((2 / np.pi) * (np.ceil(np.log(T_len) * 0.05) - 1 / np.log(T_len)), 0.1)
        ki = np.percentile(scores, 99)

        for gamma_raw in gammas_raw:
            # Normalize gamma to c = gamma / lambda_max
            c_normalized = gamma_raw / lambda_max
            
            if verbose:
                print(f"\n  {'─'*60}")
                print(f"  γ_raw = {gamma_raw:.4f}  →  c = γ/λ_max = {c_normalized:.4f}")
                print(f"  {'─'*60}")

            # ============================================================
            # STEP 3: Run ACI (Baseline)
            # ============================================================
            aci_config = read_yaml(f'config/{task}-aci-{dataset}.yaml')
            aci_config['gamma'] = gamma_raw
            
            exp_aci = ForecastingExperiment(aci_config)
            exp_aci.run()
            aci_var, aci_miscov, aci_len, aci_frac_inf, aci_std = compute_metrics(exp_aci.result, task)

            if verbose:
                print(f"  ACI: miscov={aci_miscov*100:.2f}%, std={aci_std*100:.2f}%, len={aci_len:.4f}")

            # ============================================================
            # STEP 4: Tune PID to match ACI variance
            # ============================================================
            B = np.percentile(scores, 99)
            bound_5B = 5 * B
            lo, hi = 1e-6, bound_5B
            best_eta, best_diff, best_q = lo, 1e10, None

            for _ in range(25):  # Binary search
                mid = np.sqrt(lo * hi)
                try:
                    pid_out = quantile_integrator_log(
                        scores, alpha=ALPHA, lr=mid, 
                        Csat=csat, KI=ki, ahead=1, T_burnin=100
                    )
                    q = np.maximum(np.array(pid_out['q']), 0.0)
                    
                    # Construct PID prediction intervals
                    df_temp = exp_fixed.result.copy()
                    df_temp['upper'] = y_pred_center + q
                    df_temp['lower'] = y_pred_center - q
                    df_temp['alpha'] = ALPHA
                    df_temp['beta'] = (
                        (y_true >= (y_pred_center - q)) & 
                        (y_true <= (y_pred_center + q))
                    ).astype(int)
                    
                    var_temp, _, _, _, _ = compute_metrics(df_temp, task)
                    diff = abs(var_temp - aci_var)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_eta = mid
                        best_q = q
                    
                    # Update search bounds
                    if var_temp > aci_var:
                        lo = mid
                    else:
                        hi = mid
                except:
                    hi = mid

            # Finalize PID results
            eta_found = best_eta
            df_pid = exp_fixed.result.copy()
            df_pid['upper'] = y_pred_center + best_q
            df_pid['lower'] = y_pred_center - best_q
            df_pid['alpha'] = ALPHA
            df_pid['beta'] = (
                (y_true >= (y_pred_center - best_q)) & 
                (y_true <= (y_pred_center + best_q))
            ).astype(int)
            
            pid_var, pid_miscov, pid_len, pid_frac_inf, pid_std = compute_metrics(df_pid, task)
            pid_init_miscov = compute_initial_miscov(df_pid)

            if verbose:
                print(f"  PID: η={eta_found:.6f}, var_error={(abs(pid_var-aci_var)/max(aci_var,1e-12)*100):.1f}%")
                print(f"       miscov={pid_miscov*100:.2f}%, std={pid_std*100:.2f}%, len={pid_len:.4f}")
                print(f"       init_miscov={pid_init_miscov*100:.2f}%")

            # ============================================================
            # STEP 5: Tune BCI with normalized c and lambda_init
            # ============================================================
            bci_config = read_yaml(f'config/{task}-bci-{dataset}.yaml')
            
            # Use normalized c
            bci_config['gamma'] = c_normalized * lambda_max
            
            # Tune lambda_init to match PID's initial miscoverage
            bci_config['lambda_init'] = tune_lambda_init(
                bci_config, 
                target_miscov=pid_init_miscov
            )
            
            if verbose:
                print(f"  BCI: c={c_normalized:.4f}, γ={bci_config['gamma']:.4f}")
                print(f"       λ_init={bci_config['lambda_init']:.4f} (tuned to PID init)")

            # Run BCI
            exp_bci = ForecastingExperiment(bci_config)
            exp_bci.run()
            bci_var, bci_miscov, bci_len, bci_frac_inf, bci_std = compute_metrics(exp_bci.result, task)

            if verbose:
                print(f"       miscov={bci_miscov*100:.2f}%, std={bci_std*100:.2f}%, len={bci_len:.4f}")

            # ============================================================
            # STEP 6: Generate Main Body Figures (Publication Quality)
            # ============================================================
            if save_plots:
                fig, axs = plt.subplots(2, 1, figsize=(8, 12))
                ma_window, skip = 250, 300

                def clean_df(df):
                    for col in ['upper', 'lower', 'true_y']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df

                # Generate plot data with moving average
                xform = lambda x: x
                p_aci = gen_plot_data(clean_df(exp_aci.result.copy()), ma_window, skip, xform)
                p_bci = gen_plot_data(clean_df(exp_bci.result.copy()), ma_window, skip, xform)
                p_fix = gen_plot_data(clean_df(exp_fixed.result.copy()), ma_window, skip, xform)
                p_pid = gen_plot_data(clean_df(df_pid.copy()), ma_window, skip, xform)

                # Panel 1: Miscoverage Rate
                axs[0].plot(100*p_aci['miscovrate'], color='navy', label='ACI', linewidth=3)
                axs[0].plot(100*p_bci['miscovrate'], color='tomato', label='BCI', linewidth=3)
                axs[0].plot(100*p_fix['miscovrate'], color='silver', label='Fixed', linewidth=3)
                axs[0].plot(100*p_pid['miscovrate'], color='green', label='PID', linestyle='--', linewidth=3)
                axs[0].axhline(10, color='black', linestyle='--')
                axs[0].set_ylabel('Mis-coverage rate (%)', fontsize=17)
                axs[0].legend(loc='upper right', prop={'size': 17})
                axs[0].set_xticks(p_aci['indices'])
                axs[0].set_xticklabels(
                    [p_aci['date_indices'][i] for i in p_aci['indices']], 
                    rotation=30, fontsize=17
                )
                for lbl in axs[0].get_yticklabels(): 
                    lbl.set_fontsize(17)

                # Panel 2: Interval Length
                axs[1].plot(p_aci['length'], color='navy', label='ACI', linewidth=3)
                axs[1].plot(p_bci['length'], color='tomato', label='BCI', linewidth=3)
                axs[1].plot(p_fix['length'], color='silver', label='Fixed', linewidth=3)
                axs[1].plot(p_pid['length'], color='green', label='PID', linewidth=3)
                axs[1].set_ylabel('Interval Length', fontsize=17)
                axs[1].legend(loc='upper right', prop={'size': 17})
                axs[1].set_xticks(p_aci['indices'])
                axs[1].set_xticklabels(
                    [p_aci['date_indices'][i] for i in p_aci['indices']], 
                    rotation=30, fontsize=17
                )
                for lbl in axs[1].get_yticklabels(): 
                    lbl.set_fontsize(17)

                # Title
                fig.suptitle(
                    f'Return forecasting for {dataset} (c={c_normalized:.4f})',
                    fontsize=20, y=0.98
                )
                
                plt.tight_layout()
                fname = f'{results_dir}/{dataset}_returns_c{c_normalized:.4f}_tuned.png'
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close()
                
                if verbose:
                    print(f"  ✓ Saved main body figure: {fname}")
            
            # ============================================================
            # STEP 7: Collect Results for Summary Tables
            # ============================================================
            all_rows.append({
                'dataset': dataset,
                'gamma_raw': gamma_raw,
                'c_normalized': c_normalized,
                'lambda_max': lambda_max,
                'label': f'Return-{dataset}',
                'config_label': f'c={c_normalized:.3f}',
                
                # BCI metrics
                'bci_miscov': bci_miscov * 100,
                'bci_std': bci_std * 100,
                'bci_len': bci_len,
                'bci_frac_inf': bci_frac_inf * 100,
                
                # ACI metrics
                'aci_miscov': aci_miscov * 100,
                'aci_std': aci_std * 100,
                'aci_len': aci_len,
                'aci_frac_inf': aci_frac_inf * 100,
                
                # PID metrics
                'pid_miscov': pid_miscov * 100,
                'pid_std': pid_std * 100,
                'pid_len': pid_len,
                'pid_frac_inf': pid_frac_inf * 100,
                
                # Tuning details (for reproducibility)
                'pid_eta': eta_found,
                'bci_lambda_init': bci_config['lambda_init'],
                'bci_gamma': bci_config['gamma'],
            })

    # ====================================================================
    # STEP 8: Print Summary Tables (Paper Format)
    # ====================================================================
    if verbose:
        print_paper_tables(all_rows)
    
    # Save results to CSV for later use
    df_results = pd.DataFrame(all_rows)
    df_results.to_csv(f'{results_dir}/returns_main_results.csv', index=False)
    print(f"\n✓ Saved results to {results_dir}/returns_main_results.csv")

    return all_rows


def print_paper_tables(all_rows):
    """Print formatted tables matching paper style"""
    
    print(f"\n{'='*140}")
    print("MAIN BODY TABLE — Returns Forecasting")
    print(f"{'='*140}")
    print(f"{'Dataset':<25} {'Miscoverage rate (%)':<45} {'Average length':<45} {'Frac. days with ∞':<25}")
    print(f"{'':25} {'BCI':<14} {'ACI':<14} {'PID':<14} {'BCI':<14} {'ACI':<14} {'PID':<14} {'BCI':<11} {'ACI':<11}")
    print(f"{'-'*140}")
    
    for r in all_rows:
        print(
            f"{r['label']:<25} "
            f"{r['bci_miscov']:>6.2f}%       "
            f"{r['aci_miscov']:>6.2f}%       "
            f"{r['pid_miscov']:>6.2f}%       "
            f"{r['bci_len']:>12.4f}  "
            f"{r['aci_len']:>12.4f}  "
            f"{r['pid_len']:>12.4f}  "
            f"{r['bci_frac_inf']:>9.1f}%  "
            f"{r['aci_frac_inf']:>9.1f}%"
        )
    
    print(f"{'='*140}\n")
    
    # Appendix table format
    print(f"{'='*95}")
    print("APPENDIX TABLE — Returns Forecasting (Miscoverage ± Std)")
    print(f"{'='*95}")
    print(f"{'Task and dataset':<30} {'Miscoverage rate (%) ± Std':<65}")
    print(f"{'':30} {'BCI':<21} {'ACI':<21} {'PID':<21}")
    print(f"{'-'*95}")
    
    for r in all_rows:
        print(
            f"{r['label']:<30} "
            f"{r['bci_miscov']:>5.2f}% ± {r['bci_std']:>4.2f}%   "
            f"{r['aci_miscov']:>5.2f}% ± {r['aci_std']:>4.2f}%   "
            f"{r['pid_miscov']:>5.2f}% ± {r['pid_std']:>4.2f}%"
        )
    
    print(f"{'='*95}\n")
    
    # Tuning summary (for methods section / reproducibility)
    print(f"{'='*110}")
    print("TUNING SUMMARY (for Methods/Reproducibility)")
    print(f"{'='*110}")
    print(f"{'Dataset':<15} {'c':<8} {'γ (BCI)':<12} {'λ_max':<10} {'λ_init':<10} {'η (PID)':<12}")
    print(f"{'-'*110}")
    
    for r in all_rows:
        print(
            f"{r['dataset']:<15} "
            f"{r['c_normalized']:<8.4f} "
            f"{r['bci_gamma']:<12.6f} "
            f"{r['lambda_max']:<10.4f} "
            f"{r['bci_lambda_init']:<10.4f} "
            f"{r['pid_eta']:<12.6f}"
        )
    
    print(f"{'='*110}")
    print("\n✓ Returns experiments complete!")
    print(f"  Using tuned c = γ/λ_max (normalized ratio ∈ [0,1])")
    print(f"  λ_max from {WARMUP_PERIODS}-period warmup ({PERCENTILE}th percentile)")
    print(f"  Fair comparison: variance-matched PID, init-matched BCI\n")


if __name__ == '__main__':
    # Can run standalone
    results = run_returns_experiments(save_plots=True, verbose=True)
    
    print("\n" + "="*70)
    print("NOTE: For sensitivity analysis (Appendix):")
    print("  Run: python run_sensitivity_analysis.py")
    print("="*70)