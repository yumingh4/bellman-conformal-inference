"""
Google Trends Calibrated Experiments
Two-stage tuning: gamma for variance, lambda_init for initial miscoverage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_yaml
from experiment import ForecastingExperiment
from bci_utils import compute_metrics, compute_initial_miscov, tune_lambda_init
from visualize import gen_plot_data
from pid_external.pid_methods import quantile_integrator_log


def clean_config(config):
    """Ensure all list parameters become scalars"""
    new_config = config.copy()
    for key, value in new_config.items():
        if isinstance(value, list):
            new_config[key] = value[0]
    return new_config


def estimate_lambda_max(dataset='google_trends', warmup=100):
    """Estimate lambda_max from first warmup observations"""
    cfg = read_yaml('config/trend-bci.yaml')
    cfg = clean_config(cfg)
    cfg['id'] = dataset
    exp = ForecastingExperiment(cfg)
    exp.run()
    lambdas = pd.to_numeric(exp.result['lambda'], errors='coerce').dropna()
    if len(lambdas) == 0:
        return 1.0
    return max(np.percentile(lambdas.values[:warmup], 99), 1.0)


def binary_search_bci_gamma(base_config, target_var):
    """Stage 1: Find gamma that matches ACI variance"""
    lo, hi = 1.0, 5000.0
    best_gamma, best_diff, best_result = lo, float('inf'), None

    for _ in range(30):
        mid = np.sqrt(lo * hi)
        cfg = clean_config(base_config.copy())
        cfg['gamma'] = float(mid)
        
        try:
            exp = ForecastingExperiment(cfg)
            exp.run()
            var, _, _ = compute_metrics(exp.result, 'trend')
            diff = abs(var - target_var)
            
            if diff < best_diff:
                best_diff = diff
                best_gamma = mid
                best_result = exp.result
            
            if var > target_var:
                lo = mid
            else:
                hi = mid
        except:
            hi = mid
    
    return best_gamma, best_result


def run_trends_experiments():
    """Run Google Trends experiment with two-stage tuning"""
    
    dataset = 'google_trends'
    task = 'trend'
    base_gamma = 0.1
    
    print(f"\n{'='*60}\nGOOGLE TRENDS\n{'='*60}")

    # --- Fixed baseline ---
    cfg_fix = read_yaml('config/trend-fixed.yaml')
    cfg_fix = clean_config(cfg_fix)
    cfg_fix['id'] = dataset
    exp_fixed = ForecastingExperiment(cfg_fix)
    exp_fixed.run()
    
    # --- ACI ---
    cfg_aci = read_yaml('config/trend-aci.yaml')
    cfg_aci = clean_config(cfg_aci)
    cfg_aci['id'] = dataset
    cfg_aci['gamma'] = base_gamma
    exp_aci = ForecastingExperiment(cfg_aci)
    exp_aci.run()
    aci_var, aci_miscov, aci_len = compute_metrics(exp_aci.result, task)
    
    # --- PID (binary search for eta to match ACI variance) ---
    y_true = pd.to_numeric(exp_fixed.result['true_y'], errors='coerce').values
    u_f = pd.to_numeric(exp_fixed.result['upper'], errors='coerce').values
    l_f = pd.to_numeric(exp_fixed.result['lower'], errors='coerce').values
    y_center = (u_f + l_f) / 2
    scores = np.abs(y_true - y_center)
    
    T_len = len(scores)
    ki = np.percentile(scores, 99)
    csat = max((2 / np.pi) * (np.ceil(np.log(T_len) * 0.05) - 1 / np.log(T_len)), 0.1)
    
    B = np.percentile(scores, 99)
    bound_5B = 5 * B
    lo, hi = 1e-6, bound_5B
    best_eta, best_diff, best_q = lo, 1e10, None
    
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        try:
            pid_out = quantile_integrator_log(scores, alpha=0.1, lr=mid, Csat=csat, KI=ki, ahead=1, T_burnin=100)
            q = np.maximum(np.array(pid_out['q']), 0.0)
            
            df_temp = exp_fixed.result.copy()
            df_temp['upper'] = y_center + q
            df_temp['lower'] = y_center - q
            df_temp['alpha'] = 0.1
            df_temp['beta'] = ((y_true >= (y_center - q)) & (y_true <= (y_center + q))).astype(int)
            
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
    df_pid = exp_fixed.result.copy()
    df_pid['upper'] = y_center + best_q
    df_pid['lower'] = y_center - best_q
    df_pid['alpha'] = 0.1
    df_pid['beta'] = ((y_true >= (y_center - best_q)) & (y_true <= (y_center + best_q))).astype(int)
    pid_var, pid_miscov, pid_len = compute_metrics(df_pid, task)
    
    print(f"  η = {eta_found:.4f}  |  PID var error: {abs(pid_var - aci_var) / max(aci_var, 1e-12) * 100:.1f}%")
    
    # --- BCI: Two-stage tuning ---
    print(f"  Stage 1: tuning BCI gamma to match ACI variance...")
    l_max = estimate_lambda_max(dataset, warmup=100)
    
    bci_base = read_yaml('config/trend-bci.yaml')
    bci_base = clean_config(bci_base)
    bci_base['id'] = dataset
    bci_base['lambda_max'] = float(l_max)
    
    gamma_bci, _ = binary_search_bci_gamma(bci_base, aci_var)
    print(f"    gamma → {gamma_bci:.2f}")
    
    # Stage 2: Tune lambda_init to match PID initial miscoverage
    pid_init_miscov = compute_initial_miscov(df_pid)
    print(f"  Stage 2: tuning BCI lambda_init to match PID initial miscov ({pid_init_miscov*100:.2f}%)...")
    
    bci_base['gamma'] = float(gamma_bci)
    lambda_bci = tune_lambda_init(bci_base, target_miscov=pid_init_miscov)
    print(f"    lambda_init → {lambda_bci:.0f}")
    
    # Final BCI run
    bci_base['lambda_init'] = float(lambda_bci)
    exp_bci = ForecastingExperiment(bci_base)
    exp_bci.run()
    bci_var, bci_miscov, bci_len = compute_metrics(exp_bci.result, task)
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    ma, sk = 150, 100
    
    def clean_df(df):
        for col in ['upper', 'lower', 'true_y']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    p_a = gen_plot_data(clean_df(exp_aci.result.copy()), ma, sk, lambda x: x)
    p_b = gen_plot_data(clean_df(exp_bci.result.copy()), ma, sk, lambda x: x)
    p_f = gen_plot_data(clean_df(exp_fixed.result.copy()), ma, sk, lambda x: x)
    p_p = gen_plot_data(clean_df(df_pid.copy()), ma, sk, lambda x: x)
    
    ax1.plot(100*p_a['miscovrate'], color='navy', label='ACI', linewidth=3)
    ax1.plot(100*p_b['miscovrate'], color='tomato', label='BCI', linewidth=3)
    ax1.plot(100*p_f['miscovrate'], color='silver', label='Fixed', linewidth=3)
    ax1.plot(100*p_p['miscovrate'], color='green', label='PID', linestyle='--', linewidth=3)
    ax1.axhline(10, color='black', linestyle='--')
    ax1.set_ylabel('Mis-coverage rate (%)', fontsize=17)
    ax1.legend(prop={'size': 17}, loc='upper right')
    ax1.set_xticks(p_a['indices'])
    ax1.set_xticklabels([p_a['date_indices'][i] for i in p_a['indices']], rotation=30, fontsize=17)
    for label in ax1.get_yticklabels():
        label.set_fontsize(17)
    
    ax2.plot(p_a['length'], color='navy', label='ACI', linewidth=3)
    ax2.plot(p_b['length'], color='tomato', label='BCI', linewidth=3)
    ax2.plot(p_f['length'], color='silver', label='Fixed', linewidth=3)
    ax2.plot(p_p['length'], color='green', label='PID', linewidth=3)
    ax2.set_ylabel('Interval Length', fontsize=17)
    ax2.legend(prop={'size': 17}, loc='upper right')
    ax2.set_xticks(p_a['indices'])
    ax2.set_xticklabels([p_a['date_indices'][i] for i in p_a['indices']], rotation=30, fontsize=17)
    for label in ax2.get_yticklabels():
        label.set_fontsize(17)
    
    fig.suptitle('Google Trend Popularity', fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(f'{dataset}_{task}_calibrated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {dataset}_{task}_calibrated.png")
    
    # Print summary
    print(f"\n{'='*120}")
    print("FINAL SUMMARY — GOOGLE TRENDS")
    print(f"{'='*120}")
    print(f"{'Method':<15}  Miscoverage      Std          Length")
    print("-" * 60)
    print(f"{'ACI':<15}  {aci_miscov*100:>6.2f}%      {np.sqrt(aci_var)*100:>6.2f}%      {aci_len:>8.4f}")
    print(f"{'BCI':<15}  {bci_miscov*100:>6.2f}%      {np.sqrt(bci_var)*100:>6.2f}%      {bci_len:>8.4f}")
    print(f"{'PID':<15}  {pid_miscov*100:>6.2f}%      {np.sqrt(pid_var)*100:>6.2f}%      {pid_len:>8.4f}")
    print(f"{'='*120}")
    print("✓ Google Trends experiments complete!")
    
    return {
        'aci': (aci_var, aci_miscov, aci_len),
        'bci': (bci_var, bci_miscov, bci_len),
        'pid': (pid_var, pid_miscov, pid_len)
    }


if __name__ == '__main__':
    run_trends_experiments()