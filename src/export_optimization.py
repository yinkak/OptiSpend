import pandas as pd
import numpy as np
import xarray as xr
from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper

# 1. Load your trained model and your cleaned data
mmm = MMM.load("models/mmm_model_v1_multi.nc")
data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
data["Week"] = pd.to_datetime(data["Week"])

channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]
geo_list = sorted(data["Geo"].unique())

# 2. Setup the Optimizer Wrapper
# We use a date range the model has never seen (next 8 weeks)
start_date = data["Week"].max() + pd.Timedelta(weeks=1)
wrapper = MultiDimensionalBudgetOptimizerWrapper(
    mmm, 
    start_date=start_date, 
    end_date=start_date + pd.Timedelta(weeks=8)
)

# 3. Define Constraints (Bounds)
# We set $1k min and $60k max per cell to give the AI "room to move"
num_geos, num_channels = len(geo_list), len(channels)
bounds_values = np.zeros((num_geos, num_channels, 2))
bounds_values[:, :, 0] = 1000.0  # Lower bound
bounds_values[:, :, 1] = 60000.0 # Upper bound

bounds_da = xr.DataArray(
    bounds_values,
    dims=["Geo", "channel", "bound"],
    coords={"Geo": geo_list, "channel": channels, "bound": ["lower", "upper"]}
)

# 4. RUN THE ACTUAL OPTIMIZATION
print("🚀 AI is calculating the optimal media mix...")

# We optimize for the 'original scale' so the results are in $ USD
optimal, res = wrapper.optimize_budget(
    budget=1000000.0,
    budget_bounds=bounds_da,
    response_variable="total_media_contribution_original_scale",
    # We use a randomized initial guess to force the solver to search the space
    initial_guess=None 
)

# 5. Save the 'AI-Found' values to your Demo CSV
# This converts the multidimensional array into a flat CSV recruiters can read
opt_df = optimal.to_dataframe(name="Spend").reset_index()
opt_df.to_csv("data/processed/demo_optimal_spend.csv", index=False)

print("✅ Success! The CSV now contains the model's actual optimized values.")