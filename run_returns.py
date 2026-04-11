"""
Returns (rtfc) Calibrated Experiments
Runs Amazon, AMD, Nvidia with lambda_init tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_yaml
from experiment import ForecastingExperiment
from bci_utils import compute_metrics, compute_initial_miscov, tune_lambda_init
from visualize import gen_plot_data
from pid_external.pid_methods import quantile_integrator_log


def run_returns_experiments():
    """Run all returns experiments with calibrated parameters"""
    
    datasets = ['Amazon', 'AMD', 'Nvidia']
    task     = 'rtfc'
    gammas   = [0.1, 0.008]
    all_rows = []

    for dataset in datasets:
        print(f"\n{'='*60}\n{dataset.upper()} — RETURNS\n{'='*60}")

        # Fixed baseline
        exp_fixed = ForecastingExperiment(read_yaml(f'config/{task}-fixed-{dataset}.yaml'))
        exp_fixed.run()

        # Prepare PID inputs
        y_true        = pd.to_numeric(exp_fixed.result['true_y'], errors='coerce').values
        upper         = pd.to_numeric(exp_fixed.result['upper'],  errors='coerce').values
        lower         = pd.to_numeric(exp_fixed.result['lower'],  errors='coerce').values
        y_pred_center = (upper + lower) / 2
        scores        = np.abs(y_true - y_pred_center)

        T_len = len(scores)
        csat  = max((2 / np.pi) * (np.ceil(np.log(T_len) * 0.05) - 1 / np.log(T_len)), 0.1)
        ki    = np.percentile(scores, 99)

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

            # --- PID (binary search for eta to match ACI variance) ---
            B = np.percentile(scores, 99)
            bound_5B = 5 * B
            lo, hi = 1e-6, bound_5B
            best_eta, best_diff, best_q = lo, 1e10, None

            for _ in range(25):
                mid = np.sqrt(lo * hi)
                try:
                    pid_out = quantile_integrator_log(scores, alpha=0.1, lr=mid, Csat=csat, KI=ki, ahead=1, T_burnin=100)
                    q = np.maximum(np.array(pid_out['q']), 0.0)
                    
                    df_temp = exp_fixed.result.copy()
                    df_temp['upper'] = y_pred_center + q
                    df_temp['lower'] = y_pred_center - q
                    df_temp['alpha'] = 0.1
                    df_temp['beta'] = ((y_true >= (y_pred_center - q)) & (y_true <= (y_pred_center + q))).astype(int)
                    
                    var_temp, _, _ = compute_metrics(df_temp, task)
                    diff = abs(var_temp - aci_var)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_eta = mid
                        best_q = q
                    
                    if var_temp > aci_var:
                        lo = mid
                    else:
                        hi = mid
                except:
                    hi = mid

            eta_found = best_eta
            pid_df = exp_fixed.result.copy()
            pid_df['upper'] = y_pred_center + best_q
            pid_df['lower'] = y_pred_center - best_q
            pid_df['alpha'] = 0.1
            pid_df['beta'] = ((y_true >= (y_pred_center - best_q)) & (y_true <= (y_pred_center + best_q))).astype(int)
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
            ma_window, skip = 250, 300
            xform = lambda x: x

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
            axs[1].set_ylabel('Interval Length', fontsize=17)
            axs[1].legend(loc='upper right', prop={'size': 17})
            axs[1].set_xticks(p_aci['indices'])
            axs[1].set_xticklabels([p_aci['date_indices'][i] for i in p_aci['indices']], rotation=30, fontsize=17)
            for lbl in axs[1].get_yticklabels(): lbl.set_fontsize(17)

            fig.suptitle(f'Return forecasting for {dataset} (γ={gamma})', fontsize=20, y=0.98)
            plt.tight_layout()
            fname = f'{dataset}_{task}_gamma{gamma}_calibrated.png'
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved: {fname}")

            # Collect metrics
            all_rows.append({
                'label':      f'Return-{dataset} (γ={gamma})',
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
    print("FINAL SUMMARY — RETURNS (rtfc)")
    print(f"{'='*120}")
    for r in all_rows:
        print(f"{r['label']:<35}  "
              f"{r['bci_miscov']:.2f}±{r['bci_std']:.2f}%  "
              f"{r['aci_miscov']:.2f}±{r['aci_std']:.2f}%  "
              f"{r['pid_miscov']:.2f}±{r['pid_std']:.2f}%  "
              f"Len: {r['bci_len']:.4f}  {r['aci_len']:.4f}  {r['pid_len']:.4f}")
    print(f"{'='*120}")
    print("✓ Returns experiments complete!")
    
    return all_rows


if __name__ == '__main__':
    # Can run standalone
    run_returns_experiments()