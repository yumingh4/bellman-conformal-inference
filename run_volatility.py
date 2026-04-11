"""
Volatility (vlfc) Calibrated Experiments
Runs Amazon, AMD, Nvidia with lambda_init tuning in sqrt-transformed space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_yaml
from experiment import ForecastingExperiment
from bci_utils import compute_metrics, compute_initial_miscov, tune_lambda_init
from visualize import gen_plot_data
from pid_external.pid_methods import quantile_integrator_log


def run_volatility_experiments():
    """Run all volatility experiments with calibrated parameters"""
    
    datasets = ['Amazon', 'AMD', 'Nvidia']
    task     = 'vlfc'
    gammas   = [0.1, 0.008]
    all_rows = []

    for dataset in datasets:
        print(f"\n{'='*60}\n{dataset.upper()} — VOLATILITY\n{'='*60}")

        # Fixed baseline
        exp_fixed = ForecastingExperiment(read_yaml(f'config/{task}-fixed-{dataset}.yaml'))
        exp_fixed.run()

        # Prepare PID inputs (sqrt-transformed space)
        y_true = pd.to_numeric(exp_fixed.result['true_y'], errors='coerce').values
        upper  = pd.to_numeric(exp_fixed.result['upper'],  errors='coerce').values
        lower  = pd.to_numeric(exp_fixed.result['lower'],  errors='coerce').values
        
        y_true_sqrt        = np.sqrt(np.maximum(y_true, 0))
        upper_sqrt         = np.sqrt(np.maximum(upper,  0))
        lower_sqrt         = np.sqrt(np.maximum(lower,  0))
        y_pred_center_sqrt = (upper_sqrt + lower_sqrt) / 2
        scores_sqrt        = np.abs(y_true_sqrt - y_pred_center_sqrt)

        T_len = len(scores_sqrt)
        csat  = max((2 / np.pi) * (np.ceil(np.log(T_len) * 0.05) - 1 / np.log(T_len)), 0.1)
        ki    = np.percentile(scores_sqrt, 99)

        for gamma in gammas:
            print(f"\n  γ = {gamma}")

            # --- ACI ---
            aci_config          = read_yaml(f'config/{task}-aci-{dataset}.yaml')
            aci_config['gamma'] = gamma
            exp_aci             = ForecastingExperiment(aci_config)
            exp_aci.run()
            aci_var, aci_miscov, aci_len = compute_metrics(exp_aci.result, task)

            # --- BCI (temporary, for variance target) ---
            bci_config = read_yaml(f'config/{task}-bci-{dataset}.yaml')
            if gamma == 0.008:
                bci_config['gamma']       /= 12.5
                bci_config['lambda_init'] /= 12.5

            exp_bci_tmp = ForecastingExperiment(bci_config)
            exp_bci_tmp.run()
            bci_var, _, _ = compute_metrics(exp_bci_tmp.result, task)

            # --- PID (binary search for eta to match ACI variance, sqrt space) ---
            B = np.percentile(scores_sqrt, 99)
            bound_5B = 5 * B
            lo, hi = 1e-6, bound_5B
            best_eta, best_diff, best_result = lo, 1e10, None

            for _ in range(30):
                mid = np.sqrt(lo * hi)
                try:
                    pid_out = quantile_integrator_log(
                        scores=scores_sqrt, alpha=0.1, lr=mid,
                        Csat=csat, KI=ki, ahead=1, T_burnin=100
                    )
                    q = np.maximum(np.array(pid_out['q']), 0.0)
                    
                    upper_sqrt_pid = y_pred_center_sqrt + q
                    lower_sqrt_pid = np.maximum(y_pred_center_sqrt - q, 0)
                    upper_orig = upper_sqrt_pid ** 2
                    lower_orig = lower_sqrt_pid ** 2
                    
                    df_temp = exp_fixed.result.copy()
                    df_temp['upper'] = upper_orig
                    df_temp['lower'] = lower_orig
                    df_temp['alpha'] = 0.1
                    df_temp['beta'] = ((y_true >= lower_orig) & (y_true <= upper_orig)).astype(int)
                    
                    var_temp, _, _ = compute_metrics(df_temp, task)
                    diff = abs(var_temp - aci_var)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_eta = mid
                        best_result = (q, upper_orig, lower_orig)
                    
                    if var_temp > aci_var:
                        lo = mid
                    else:
                        hi = mid
                except:
                    hi = mid

            if best_result is None:
                print(f"    ✗ PID binary search failed for γ={gamma}")
                continue

            eta_found = best_eta
            pid_q, upper_orig, lower_orig = best_result
            
            pid_df = exp_fixed.result.copy()
            pid_df['upper'] = upper_orig
            pid_df['lower'] = lower_orig
            pid_df['alpha'] = 0.1
            pid_df['beta'] = ((y_true >= lower_orig) & (y_true <= upper_orig)).astype(int)
            pid_var, pid_miscov, pid_len = compute_metrics(pid_df, task)

            print(f"    η = {eta_found:.4f}  |  PID var error: {abs(pid_var - aci_var) / max(aci_var, 1e-12) * 100:.1f}%")

            # --- BCI: tune lambda_init to match PID's initial miscoverage ---
            pid_init_miscov = compute_initial_miscov(pid_df)
            print(f"    PID initial miscov: {pid_init_miscov*100:.2f}% → tuning BCI lambda_init...")
            
            bci_config['lambda_init'] = tune_lambda_init(bci_config, target_miscov=pid_init_miscov)
            print(f"    lambda_init → {bci_config['lambda_init']:.4f}")

            exp_bci = ForecastingExperiment(bci_config)
            exp_bci.run()
            bci_var, bci_miscov, bci_len = compute_metrics(exp_bci.result, task)

            # --- Plot ---
            fig, axs = plt.subplots(2, 1, figsize=(8, 12))
            ma_window, skip = 250, 500
            xform = lambda x: np.sqrt(np.maximum(x, 0))

            def clean_df(df):
                for col in ['upper', 'lower', 'true_y']:
                    if col in df.columns: 
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df

            p_aci = gen_plot_data(clean_df(exp_aci.result.copy()),   ma_window, skip, xform)
            p_bci = gen_plot_data(clean_df(exp_bci.result.copy()),   ma_window, skip, xform)
            p_fix = gen_plot_data(clean_df(exp_fixed.result.copy()), ma_window, skip, xform)
            p_pid = gen_plot_data(clean_df(pid_df.copy()),           ma_window, skip, xform)

            axs[0].plot(100*p_aci['miscovrate'], color='navy',   label='ACI',    linewidth=3)
            axs[0].plot(100*p_bci['miscovrate'], color='tomato', label='BCI',    linewidth=3)
            axs[0].plot(100*p_fix['miscovrate'], color='silver', label='Fixed',  linewidth=3)
            axs[0].plot(100*p_pid['miscovrate'], color='green',  label='PID', linestyle='--', linewidth=3)
            axs[0].axhline(10, color='black', linestyle='--')
            axs[0].set_ylabel('Mis-coverage rate (%)', fontsize=17)
            axs[0].legend(loc='upper right', prop={'size': 17})
            axs[0].set_xticks(p_aci['indices'])
            axs[0].set_xticklabels([p_aci['date_indices'][i] for i in p_aci['indices']], rotation=30, fontsize=17)
            for lbl in axs[0].get_yticklabels(): lbl.set_fontsize(17)

            axs[1].plot(p_aci['length'], color='navy',   label='ACI',   linewidth=3)
            axs[1].plot(p_bci['length'], color='tomato', label='BCI',   linewidth=3)
            axs[1].plot(p_fix['length'], color='silver', label='Fixed', linewidth=3)
            axs[1].plot(p_pid['length'], color='green',  label='PID',   linewidth=3)
            axs[1].set_ylabel('Interval Length (√ space)', fontsize=17)
            axs[1].legend(loc='upper right', prop={'size': 17})
            axs[1].set_xticks(p_aci['indices'])
            axs[1].set_xticklabels([p_aci['date_indices'][i] for i in p_aci['indices']], rotation=30, fontsize=17)
            for lbl in axs[1].get_yticklabels(): lbl.set_fontsize(17)

            fig.suptitle(f'Volatility forecasting for {dataset} (γ={gamma})', fontsize=20, y=0.98)
            plt.tight_layout()
            fname = f'{dataset}_{task}_gamma{gamma}_calibrated.png'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved: {fname}")

            # Collect metrics
            all_rows.append({
                'label':      f'Volatility-{dataset} (γ={gamma})',
                'bci_miscov': bci_miscov * 100,
                'aci_miscov': aci_miscov * 100,
                'pid_miscov': pid_miscov * 100,
                'bci_std':    np.sqrt(bci_var) * 100,
                'aci_std':    np.sqrt(aci_var) * 100,
                'pid_std':    np.sqrt(pid_var) * 100,
                'bci_len':    bci_len,
                'aci_len':    aci_len,
                'pid_len':    pid_len,
            })

    # Print summary table
    print(f"\n{'='*120}")
    print("FINAL SUMMARY — VOLATILITY (vlfc)")
    print(f"{'='*120}")
    for r in all_rows:
        print(f"{r['label']:<40}  "
              f"{r['bci_miscov']:.2f}±{r['bci_std']:.2f}%  "
              f"{r['aci_miscov']:.2f}±{r['aci_std']:.2f}%  "
              f"{r['pid_miscov']:.2f}±{r['pid_std']:.2f}%  "
              f"Len: {r['bci_len']:.4f}  {r['aci_len']:.4f}  {r['pid_len']:.4f}")
    print(f"{'='*120}")
    print("✓ Volatility experiments complete!")
    
    return all_rows


if __name__ == '__main__':
    run_volatility_experiments()