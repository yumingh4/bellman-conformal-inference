import glob
import pandas as pd
import os
import re

# 1. Define the folder and keywords
path = "tuning/*.csv"
all_files = glob.glob(path)
combined_data = []

datasets = ['Amazon', 'AMD', 'Nvidia', 'deeplearning']
tasks = ['vlfc', 'rtfc', 'google']

print(f"Found {len(all_files)} files. Categorizing with lambda_max...")

for file in all_files:
    if "Master" in file:
        continue
        
    df = pd.read_csv(file)
    filename = os.path.basename(file)
    
    # --- SMART SEARCH LOGIC ---
    found_dataset = "unknown"
    for d in datasets:
        if d.lower() in filename.lower():
            found_dataset = d
            break
    
    found_task = "unknown"
    for t in tasks:
        if t.lower() in filename.lower():
            found_task = t
            break
            
    gamma_match = re.search(r'gamma(0\.\d+)', filename)
    found_gamma = gamma_match.group(1) if gamma_match else "unknown"

    # Add metadata columns
    df['Dataset'] = found_dataset
    df['Task'] = found_task
    df['Gamma_Target'] = found_gamma
    
    combined_data.append(df)

# 2. Final Merge and Sort
if combined_data:
    master_df = pd.concat(combined_data, ignore_index=True)
    
    # Sort for professional presentation
    master_df = master_df.sort_values(by=['Task', 'Dataset', 'Gamma_Target', 'method', 'T'])
    
    # --- ADDED lmax, mult, AND c TO THE LIST ---
    cols = [
        'Task', 'Dataset', 'Gamma_Target', 'method', 'T', 
        'mult', 'lmax', 'c',  # <--- These are the config values you need
        'miscov', 'length', 'rolling_std', 'frac_inf'
    ]
    
    # Ensure we only pick columns that actually exist
    existing_cols = [col for col in cols if col in master_df.columns]
    master_df = master_df[existing_cols]

    master_df.to_csv('tuning/Master_BCI_Results_Organized.csv', index=False)
    print("Success! Master CSV now includes lmax and tuning parameters.")
else:
    print("No files found.")