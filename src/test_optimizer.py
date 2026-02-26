import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)

# 1) Fit a model (toy example)
X = pd.DataFrame(
    {
        "date": pd.date_range("2025-01-01", periods=30, freq="W-MON"),
        "geo": np.random.choice(["A", "B"], size=30),
        "C1": np.random.rand(30),
        "C2": np.random.rand(30),
    }
)
y = pd.Series(np.random.rand(30), name="y")

mmm = MMM(
    date_column="date",
    dims=("geo",),
    channel_columns=["C1", "C2"],
    target_column="y",
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation(),
)
mmm.fit(X, y)

# 2) Wrap the fitted model
wrapper = MultiDimensionalBudgetOptimizerWrapper(
    model=mmm,
    start_date=X["date"].max() + pd.Timedelta(weeks=1),
    end_date=X["date"].max() + pd.Timedelta(weeks=8),
)

# Define dates based on the wrapper's range
dates = pd.date_range(wrapper.start_date, wrapper.end_date, freq="W-MON")

# 3) Fix the Factors (the 3D cube)
num_dates = len(dates)
num_geos = 2
num_channels = 2

# Shape must be (date, geo, channel)
factors_data = np.full((num_dates, num_geos, num_channels), 1 / num_dates)

factors = xr.DataArray(
    factors_data,
    dims=["date", "geo", "channel"],
    coords={
        "date": np.arange(num_dates),
        "geo": ["A", "B"],
        "channel": ["C1", "C2"]
    },
)

# 4) Fix the mask dimensions
budgets_to_optimize = xr.DataArray(
    np.array([[True, True], [True, True]]),
    dims=["geo", "channel"], # Align with wrapper expectations
    coords={"geo": ["A", "B"], "channel": ["C1", "C2"]},
)

# 5) Optimize
optimal, res = wrapper.optimize_budget(
    budget=100.0,
    budgets_to_optimize=budgets_to_optimize,
    budget_distribution_over_period=factors,
    response_variable="total_media_contribution_original_scale",
)

print("\n--- OPTIMAL ALLOCATION ---")
print(optimal.to_dataframe(name="Spend"))