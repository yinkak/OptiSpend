import pandas as pd
import numpy as np
import os
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

def save_causal_report(ci_model, path, figsize=(15, 12)):
    """
    Helper function to bypass CausalImpact's restrictive plotting.
    Saves the 'original', 'pointwise', and 'cumulative' panels to a file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Use the library's internal plotting helper but capture the figure
    # This uses the logic from your template to manually build the subplots
    fig = plt.figure(figsize=figsize)
    
    # Get the data from the model
    llb = ci_model.trained_model.filter_results.loglikelihood_burn
    inferences = ci_model.inferences.iloc[llb:]
    intervention_idx = inferences.index.get_loc(ci_model.post_period[0])
    
    # --- Panel 1: Original ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(pd.concat([ci_model.pre_data.iloc[llb:, 0], ci_model.post_data.iloc[:, 0]]), 'k', label='Actual Sales')
    ax1.plot(inferences['preds'], 'b--', label='Counterfactual (Predicted)')
    ax1.axvline(inferences.index[intervention_idx - 1], c='r', linestyle='--', label='Intervention')
    ax1.fill_between(inferences.index, inferences['preds_lower'], inferences['preds_upper'], facecolor='blue', alpha=0.15)
    ax1.set_title("Causal Impact: Actual vs. Predicted")
    ax1.legend()

    # --- Panel 2: Pointwise ---
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(inferences['point_effects'], 'g--', label='Point Effect (Lift)')
    ax2.axhline(y=0, color='k', linestyle='-')
    ax2.fill_between(inferences.index, inferences['point_effects_lower'], inferences['point_effects_upper'], facecolor='green', alpha=0.15)
    ax2.legend()

    # --- Panel 3: Cumulative ---
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(inferences['post_cum_effects'], 'r-', label='Cumulative Impact')
    ax3.axhline(y=0, color='k', linestyle='-')
    ax3.fill_between(inferences.index, inferences['post_cum_effects_lower'], inferences['post_cum_effects_upper'], facecolor='red', alpha=0.15)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close() # Close to free up memory
    print(f"✅ Causal report saved to: {path}")

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
    #ci.plot()
    save_causal_report(ci, "reports/sample_forecast.png") 
    
    return ci

if __name__ == "__main__":
    run_causal_experiment()