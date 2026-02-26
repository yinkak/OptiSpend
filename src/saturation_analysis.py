import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pymc_marketing.mmm import MMM
from utils import unscale_sales  # Import your working helper

def plot_saturation_curves():
    print("Loading model and calculating curves...")
    mmm = MMM.load("models/mmm_model_v1_multi.nc")
    spend_scaler = joblib.load("models/spendscaler.joblib")
    
    # 1. Use the built-in helper to get the curve data
    # This returns the 'contribution' for different 'intensity' levels of spend
    # We use 'original' to try and get the raw scale, or unscale manually
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
    
    # Generate the plot using pymc-marketing's optimized method
    fig = mmm.plot_direct_contribution_curves()
    
    # 2. Add Business Context: Mark our "Current Average Spend"
    # This shows stakeholders WHERE on the curve they currently sit.
    raw_data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
    
    # Let's customize the plot to make it 'OptiSpend' branded
    plt.suptitle("Saturation Curves: Where is our Budget Wasted?", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig("plots/saturation_curves_raw.png")
    print("Raw saturation curves saved to plots/")

if __name__ == "__main__":
    plot_saturation_curves()