import pandas as pd
import numpy as np
import xarray as xr
from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper

# 1. Load Local Model & Data
mmm = MMM.load("models/mmm_model_v1_multi.nc")
data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]
geo_list = sorted(data["Geo"].unique())

# 2. Setup a Standard $1M Optimization
start = pd.to_datetime(data["Week"].max()) + pd.Timedelta(weeks=1)
wrapper = MultiDimensionalBudgetOptimizerWrapper(mmm, start_date=start, end_date=start + pd.Timedelta(weeks=8))

# Define generic bounds for the export
bounds_values = np.zeros((len(geo_list), len(channels), 2))
bounds_values[:, :, 0], bounds_values[:, :, 1] = 1000, 60000
bounds_da = xr.DataArray(bounds_values, dims=["Geo", "channel", "bound"], 
                         coords={"Geo": geo_list, "channel": channels, "bound": ["lower", "upper"]})

# 3. Run and Save
optimal, res = wrapper.optimize_budget(
    budget=1000000.0,
    budget_bounds=bounds_da,
    response_variable="total_media_contribution_original_scale"
)

# Export to a tiny CSV
optimal.to_dataframe(name="Spend").to_csv("data/processed/demo_optimal_spend.csv")
print("✅ Optimization export complete!")