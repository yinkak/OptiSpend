import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pymc_marketing.mmm import MMM
import os
from utils import unscale_sales


def calculate_roas(df_results, raw_data_path):
    """
    Calculates Return on Ad Spend (ROAS) by comparing attributed revenue
    to actual historical costs.
    """
    raw_data = pd.read_csv(raw_data_path)
    
    # We iterate through the revenue results we just calculated
    for i, row in df_results.iterrows():
        channel_name = row['Channel']
        total_revenue = row['Total_Sales_Contribution']
        
        # Get the actual total spend from the raw data
        total_cost = raw_data[channel_name].sum()
        
        # Calculate ROAS (Revenue / Cost)
        roas = total_revenue / total_cost if total_cost > 0 else 0
        
        df_results.at[i, 'Total_Cost'] = total_cost
        df_results.at[i, 'ROAS'] = roas
    
    return df_results


def run_reporting():
    print("Loading model...")
    loaded_mmm = MMM.load("models/mmm_model_v1.nc")
    
    # 1. Manually calculate the mean contribution from the posterior
    # This replaces the missing 'compute_channel_contribution_stats'
    print("Extracting channel contributions...")
    
    # We take the mean across chains, draws, and time (date)
    # The result is a series where index = channel names
    contributions_scaled = loaded_mmm.idata.posterior["channel_contribution"].sum(
        dim="date").mean(dim=["chain", "draw"]).to_series()
    
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
    summary_data = []
    
    for channel in channels:
        # Get the mean scaled value from our manual calculation
        scaled_mean = contributions_scaled[channel]
        
        # 2. Use your utility function to translate to Dollars
        real_dollars = unscale_sales(scaled_mean)[0][0]
        
        summary_data.append({
            "Channel": channel, 
            "Total_Sales_Contribution": real_dollars
        })

    df_results = pd.DataFrame(summary_data)

    # Extract the Intercept (Baseline Sales)
    intercept_scaled = loaded_mmm.idata.posterior["intercept"].mean(dim=["chain", "draw"])
    real_intercept = unscale_sales(intercept_scaled)[0][0]

    print(f"Total Baseline (Organic) Sales: ${real_intercept:,.2f}")

    # 1. Calculate the Business Metrics
    raw_data_path = "data/processed/cleaned_marketing_data.csv"
    df_final = calculate_roas(df_results, raw_data_path)
    
    print("\n--- Marketing Effectiveness Report ---")
    print(df_final[['Channel', 'Total_Sales_Contribution', 'Total_Cost', 'ROAS']])
    
    # 2. Visualize Efficiency (ROAS)
    plot_roas_dashboard(df_final)
    
    # 3. Visualization
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df_results['Channel'], df_results['Total_Sales_Contribution'], color="#32cd32")
    
    # Format Y-axis as Currency for stakeholders
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    plt.title("Attributed Sales Revenue per Channel", fontsize=14)
    plt.ylabel("Sales Contribution (USD)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add dollar labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'${yval:,.0f}', va='bottom', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("plots/revenue_by_channel.png")
    print("Success! Revenue report saved to plots/revenue_by_channel.png")
    
    return df_results

def plot_roas_dashboard(df):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4' if r >= 1 else '#d62728' for r in df['ROAS']]
    
    bars = plt.bar(df['Channel'], df['ROAS'], color=colors)
    plt.axhline(y=1.0, color='black', linestyle='--', label="Break-even (1.0x)")
    
    plt.title("Channel Efficiency (ROAS)", fontsize=14)
    plt.ylabel("Return on Ad Spend (Multiple)")
    plt.legend()
    
    # Label the bars with the ROAS value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("plots/roas_efficiency_report.png")
    print("Efficiency report saved to plots/roas_efficiency_report.png")


if __name__ == "__main__":
    run_reporting()