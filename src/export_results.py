from pymc_marketing.mmm.multidimensional import MMM
import pandas as pd

# Load your local heavy model
mmm = MMM.load("models/mmm_model_v1_multi.nc")

# 1. Extract Channel ROI (Betas)
roi = mmm.idata.posterior["saturation_beta"].mean(("chain", "draw", "Geo")).to_series()
roi.to_csv("data/processed/demo_roi.csv")

# 2. Extract Contribution Share
contrib = mmm.idata.posterior["channel_contribution"].sum(dim=["date", "Geo"]).median(dim=["chain", "draw"]).to_series()
contrib.to_csv("data/processed/demo_contribution.csv")