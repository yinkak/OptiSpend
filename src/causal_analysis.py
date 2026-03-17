import pandas as pd
import numpy as np
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

def run_causal_experiment(data_path="data/processed/cleaned_marketing_data.csv"):
    # --- 1. PREPARE THE DATA ---
    df = pd.read_csv(data_path)
    df['Week'] = pd.to_datetime(df['Week'])
    
    df_agg = df.groupby(['Week', 'Geo'])['Sales_Value'].sum().reset_index()
    analysis_df = df_agg.pivot(index='Week', columns='Geo', values='Sales_Value')

    analysis_df = analysis_df[['NORTH', 'SOUTH', 'WEST']]
    
    # --- 2. SIMULATE A CAMPAIGN ---
    intervention_start = analysis_df.index[70] 
    intervention_end = analysis_df.index[80] 

    # Apply the 'lift' to the North only
    analysis_df.loc[intervention_start:intervention_end, 'NORTH'] *= 1.20
    
    print(f"🚀 Simulated Campaign in NORTH from {intervention_start.date()} to {intervention_end.date()}")

    # --- 3. DEFINE THE PERIODS ---
    pre_period = [str(analysis_df.index[0].date()), str(analysis_df.index[69].date())]
    post_period = [str(intervention_start.date()), str(intervention_end.date())]

    # --- 4. RUN THE ANALYSIS ---
    ci = CausalImpact(analysis_df, pre_period, post_period)
    
    # --- 5. RESULTS & EXPLANATION ---
    print("\n--- CAUSAL ANALYSIS REPORT ---")
    print(ci.summary()) 
    ci.plot()
    plt.savefig("reports/causal_impact_test.png")
    
    return ci

if __name__ == "__main__":
    run_causal_experiment()