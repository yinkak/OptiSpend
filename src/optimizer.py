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

mmm = MMM.load("models/mmm_model_v1_multi.nc")
data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
data["Week"] = pd.to_datetime(data["Week"])
geo_list = get_distinct_geos(data)
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]

last_date = data["Week"].max()
planning_start = last_date + pd.Timedelta(weeks=1)
planning_end = planning_start + pd.Timedelta(weeks=8) # 8-week window

start_date=data["Week"].max() + pd.Timedelta(weeks=1),
end_date=data["Week"].max() + pd.Timedelta(weeks=8),

start = data["Week"].max() + pd.Timedelta(weeks=1)
# Use weeks=8 here to ensure a full 8-week duration
end = start + pd.Timedelta(weeks=8) 

print(start_date, end_date)
print(planning_end, planning_start)


# 2) Wrap the fitted model for allocation over a future window
wrapper = MultiDimensionalBudgetOptimizerWrapper(
    model=mmm,
    # start_date=data["Week"].max() + pd.Timedelta(weeks=1),
    # end_date=data["Week"].max() + pd.Timedelta(weeks=8),
    start_date=start,
    end_date=end,
)

# Optional: choose which (channel, geo) cells to optimize
budgets_to_optimize = xr.DataArray(
    # np.array([[True, False], [True, True]]),
    np.full((len(channels), len(geo_list)), True),
    dims=["channel", "Geo"],
    coords={"channel": channels, 
            "Geo": geo_list},
)

budgets_to_optimize = budgets_to_optimize.transpose("Geo", "channel")

# Optional: distribute each cell's budget over the time window (must sum to 1 along date)
dates = pd.date_range(wrapper.start_date, wrapper.end_date, freq="W-MON")

num_dates = wrapper.num_periods       # Should be 8
print("NUM DATES: ", num_dates)
num_geos = len(geo_list)    # Should be 8 (based on your previous error)
num_channels = len(channels) # Should be 4

factors = xr.DataArray(
    # np.vstack(
    #     [
    #         np.full(len(dates), 1 / len(dates)),  # C1: uniform
    #         np.linspace(0.7, 0.3, len(dates)),  # C2: front‑to‑back taper
    #     ]
    # ),

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
upper_limit = 30000.0  # Increased slightly to give the optimizer room to move

# 2. Build the 3D array: (Geo, channel, 2)
# The third dimension contains [lower, upper]
bounds_values = np.zeros((num_geos, num_channels, 2))
bounds_values[:, :, 0] = lower_limit # Set all lowers
bounds_values[:, :, 1] = upper_limit # Set all uppers

# 3. Construct as xarray following your template
bounds = xr.DataArray(
    bounds_values,
    dims=["Geo", "channel", "bound"], # "bound" is the standard name for the min/max dim
    coords={
        "Geo": geo_list,
        "channel": channels,
        "bound": ["lower", "upper"]
    },
)


# def multidimensional_response(samples):
#     #Sum across axis 2 and 3 (Geo and Channel)
#     total_sales = samples.sum(axis=(2, 3))
    
#     return total_sales.mean()

beta_values = mmm.idata.posterior["saturation_beta"].mean(("chain", "draw")).values

#TODO: DELETE
def direct_beta_utility(samples, budgets, **kwargs):
    # Convert beta_values (8x4) to a tensor that matches budgets (8x4)
    beta_tensor = pt.as_tensor_variable(beta_values)
    
    # Linear effectiveness: (Budget in West * West Beta) + ...
    # This guarantees a non-zero gradient!
    return (budgets * beta_tensor).sum()

initial_guess = np.full((num_geos, num_channels), 500000.0 / 32)
initial_guess[0,0] += 1.0 

#TODO: DELETE
def custom_utility(samples,budgets, **kwargs):
    # Sum across Geo (axis 2) and Channel (axis 3), then average the Bayesian draws
    media_impact = pt.mean(samples.sum()) * 1e6 
    cost_penalty = budgets.sum() * 0.0001
    
    return media_impact - cost_penalty

def balanced_utility(samples, budgets, **kwargs):
    beta_tensor = pt.as_tensor_variable(beta_values)
    
    # We use a log-transform or a small weight on the samples 
    # to let the model's internal saturation curves "push back"
    saturation_effect = pt.mean(samples.sum()) 
    direction_effect = (budgets * beta_tensor).sum() * 1e-4 # Subtle nudge
    
    return saturation_effect + direction_effect

print(mmm.model.named_vars.keys())

# 3) Optimize
# optimal, res = wrapper.optimize_budget(
#     budget=500000.0,
#     budgets_to_optimize=budgets_to_optimize,
#     budget_distribution_over_period=factors,
#     #bounds=(lower_bounds, upper_bounds),
#     response_variable="total_media_contribution_original_scale",
#     utility_function=custom_utility
# )

optimal, res = wrapper.optimize_budget(
    budget=1000000.0,
    budgets_to_optimize=budgets_to_optimize,
    budget_distribution_over_period=factors,
    # We still provide this, but our custom_utility will do the heavy lifting
    budget_bounds=bounds,
    response_variable="total_media_contribution_original_scale",
    utility_function=balanced_utility,
)
# `optimal` is an xr.DataArray with dims (channel, geo)

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


print("Mask Sum:", budgets_to_optimize.sum().values) # Should be 32

print(f"Iterations: {res.nit}")

#<------ ANY DIMENSIONALITY ---------->
