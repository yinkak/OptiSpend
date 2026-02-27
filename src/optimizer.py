#<-------MULTIDIMENSIONAL OPTIMIZATION--------->

import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation 
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)
from utils import get_distinct_geos
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns

mmm = MMM.load("models/mmm_model_v1_multi.nc")
data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
data["Week"] = pd.to_datetime(data["Week"])
geo_list = get_distinct_geos(data)
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]

start = data["Week"].max() + pd.Timedelta(weeks=1)
end = start + pd.Timedelta(weeks=8) 

# 2) Wrap the fitted model for allocation over a future window
wrapper = MultiDimensionalBudgetOptimizerWrapper(
    model=mmm,
    start_date=start,
    end_date=end,
)

#choose which (channel, geo) cells to optimize
budgets_to_optimize = xr.DataArray(
    np.full((len(channels), len(geo_list)), True),
    dims=["channel", "Geo"],
    coords={"channel": channels, 
            "Geo": geo_list},
)

budgets_to_optimize = budgets_to_optimize.transpose("Geo", "channel")

#distribute each cell's budget over the time window (must sum to 1 along date)
dates = pd.date_range(wrapper.start_date, wrapper.end_date, freq="W-MON")

num_dates = wrapper.num_periods       # Should be 8
print("NUM DATES: ", num_dates)
num_geos = len(geo_list)
num_channels = len(channels)

factors = xr.DataArray(
    np.full((num_dates, num_geos, num_channels), 1 / num_dates),
    dims=["date","Geo", "channel"],
    coords={
        "date": np.arange(num_dates),
        "Geo": geo_list,
        "channel": channels
    },
)

total_historical_spend = data[channels].sum().sum()
# 1. Define the limits
lower_limit = 1000.0
upper_limit = 50000.0  # Increased slightly to give the optimizer room to move

# 2. Build the 3D array: (Geo, channel, 2)
bounds_values = np.zeros((num_geos, num_channels, 2))
bounds_values[:, :, 0] = lower_limit # Set all lowers
bounds_values[:, :, 1] = upper_limit # Set all uppers

# 3. Construct as xarray
bounds = xr.DataArray(
    bounds_values,
    dims=["Geo", "channel", "bound"],
    coords={
        "Geo": geo_list,
        "channel": channels,
        "bound": ["lower", "upper"]
    },
)

beta_values = mmm.idata.posterior["saturation_beta"].mean(("chain", "draw")).values

# 2. TARGETED OVERRIDE: Increase specific caps to $60,000
# We use .loc to specify [Geo, Channel, Bound_Type]
bounds.loc[{"Geo": "NORTH", "channel": "Facebook_Spend", "bound": "upper"}] = 60000.0
bounds.loc[{"Geo": "WEST", "channel": "TV_Spend", "bound": "upper"}] = 60000.0

#TODO: DELETE
def direct_beta_utility(samples, budgets, **kwargs):
    beta_tensor = pt.as_tensor_variable(beta_values)

    return (budgets * beta_tensor).sum()

initial_guess = np.full((num_geos, num_channels), 500000.0 / 32)
initial_guess[0,0] += 1.0 

def balanced_utility(samples, budgets, **kwargs):
    beta_tensor = pt.as_tensor_variable(beta_values)
    # We use a log-transform or a small weight on the samples 
    saturation_effect = pt.mean(samples.sum()) 
    direction_effect = (budgets * beta_tensor).sum() * 1e-4 
    
    return saturation_effect + direction_effect

print(mmm.model.named_vars.keys())

# --- 1. CALCULATE THE "AI DIVIDEND" (PREDICTED LIFT) ---

def calculate_lift(mmm, optimal_spend_da, historical_data, channels, total_budget):
    # 1. Get Historical Mix (% of total spend historically per cell)
    hist_totals = historical_data.groupby("Geo")[channels].sum()
    hist_total_sum = hist_totals.values.sum()
    hist_mix_percentage = hist_totals / hist_total_sum
    
    # 2. Apply that old mix to our NEW total budget
    bau_spend_values = hist_mix_percentage.values * total_budget
    bau_spend_da = xr.DataArray(
        bau_spend_values,
        dims=["Geo", "channel"],
        coords={"Geo": hist_mix_percentage.index, "channel": channels}
    )

    # 3. Calculate Contribution using Beta ROI
    betas = mmm.idata.posterior["saturation_beta"].mean(("chain", "draw"))
    
    opt_impact = (optimal_spend_da * betas).sum().values
    bau_impact = (bau_spend_da * betas).sum().values
    
    lift = opt_impact - bau_impact
    pct_gain = (lift / bau_impact) * 100

    print("\n" + "="*40)
    print(" APPLES-TO-APPLES FINANCIAL ANALYSIS ")
    print(f" (Comparing two ${total_budget:,.0f} plans) ")
    print("="*40)
    print(f"AI Optimal Contribution:   ${opt_impact:,.2f}")
    print(f"Historical Mix Contribution: ${bau_impact:,.2f}")
    print(f"PREDICTED VALUE ADD:       ${lift:,.2f}")
    print(f"EFFICIENCY GAIN:           {pct_gain:.1f}%")
    print("="*40)
    
    return bau_spend_da

# --- 2. COMPARE OPTIMAL VS. ACTUAL (THE SPEND GAP) ---

def plot_spend_comparison(optimal_spend_da, bau_spend_da):
    opt_df = optimal_spend_da.to_dataframe(name="Spend").reset_index()
    opt_df["Scenario"] = "AI Optimal"

    bau_df = bau_spend_da.to_dataframe(name="Spend").reset_index()
    bau_df["Scenario"] = "Historical Mix"
    
    comparison_df = pd.concat([opt_df, bau_df])

    plt.figure(figsize=(14, 8))
    sns.barplot(data=comparison_df, x="Geo", y="Spend", hue="Scenario")
    plt.title("The Strategy Shift: Where AI relocates your budget", fontsize=16)
    plt.ylabel("Weekly Spend ($)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

# --- 3. VISUALIZE SATURATION (THE "WHY") ---

def plot_top_response_curves(mmm, geo_name="WEST", channel_name="TV_Spend"):
    """
    Visualizes the saturation curve for a specific cell to see diminishing returns.
    """
    from pymc_marketing.mmm.utils import estimate_menten_parameters
    
    # Extract the specific Beta and Lambda for this Geo/Channel
    beta = mmm.idata.posterior["saturation_beta"].sel(Geo=geo_name, channel=channel_name).mean().values
    lam = mmm.idata.posterior["saturation_lam"].sel(Geo=geo_name, channel=channel_name).mean().values
    
    # Generate a range of possible spends
    x = np.linspace(0, 100000, 100)
    # Michaelis-Menten / Hill Equation logic (standard in Logistic Saturation)
    y = beta * (x / (x + lam)) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='forestgreen', lw=3, label=f"Saturation Curve: {geo_name} {channel_name}")
    plt.axvline(30000, color='red', linestyle='--', label="Old Cap ($30k)")
    plt.axvline(60000, color='blue', linestyle='--', label="New Cap ($60k)")
    
    plt.title(f"Saturation Analysis: {geo_name} | {channel_name}", fontsize=14)
    plt.xlabel("Weekly Spend ($)")
    plt.ylabel("Predicted Sales Contribution")
    plt.legend()
    plt.show()

def export_media_brief(optimal_spend_da, filename="media_buying_brief.csv"):
    """
    Converts the high-dimensional plan into a simple CSV for the buying team.
    """
    # 1. Flatten the DataArray to a DataFrame
    brief_df = optimal_spend_da.to_dataframe(name="Weekly_Spend_USD").reset_index()
    
    # 2. Add a 'Percentage_of_Total' column for quick reference
    total = brief_df["Weekly_Spend_USD"].sum()
    brief_df["Budget_Share_%"] = (brief_df["Weekly_Spend_USD"] / total * 100).round(2)
    
    # 3. Sort by highest spend so they see the priorities
    brief_df = brief_df.sort_values(by="Weekly_Spend_USD", ascending=False)
    
    # 4. Save to CSV
    brief_df.to_csv(filename, index=False)
    print(f"\n✅ Media Brief exported successfully to: {filename}")
    return brief_df

# 3) Optimize
optimal, res = wrapper.optimize_budget(
    budget=1000000.0,
    budgets_to_optimize=budgets_to_optimize,
    budget_distribution_over_period=factors,
    budget_bounds=bounds,
    response_variable="total_media_contribution_original_scale",
    utility_function=balanced_utility,
)

# ---FORMATTED OUTPUT ---
print(f"\n--- OPTISPEND PLAN ({start.date()} to {end.date()}) ---")
# Convert xarray results to a readable DataFrame
opt_df = optimal.to_dataframe(name="Recommended_Spend").unstack(level=1)
print(opt_df.round(2))

print(mmm.idata.posterior["saturation_beta"].mean(("chain", "draw")).values)
print(f"Model Dims: {mmm.model.coords.keys()}")


# Check the 'Success' status of the optimizer
print(f"Optimization Success: {res.success}")
print(f"Optimization Message: {res.message}")
print(f"Number of Iterations: {res.nit}")
print("Mask Sum:", budgets_to_optimize.sum().values)
print(f"Iterations: {res.nit}")

# --- RUN THE ANALYSIS ---
if res.success:
    print("\nStarting Post-Optimization Analysis...")
    
    target_budget = 1000000.0
    
    bau_da = calculate_lift(mmm, optimal, data, channels, target_budget)
    
    plot_spend_comparison(optimal, bau_da)
    
    plot_top_response_curves(mmm, geo_name="WEST", channel_name="TV_Spend")

    brief = export_media_brief(optimal)
    print(brief.head(10))
else:
    print("Optimization failed, skipping analysis.")




