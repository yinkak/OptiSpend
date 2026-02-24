import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import MaxAbsScaler
from mmm_model import build_mmm 
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import matplotlib.pyplot as plt
import os

#TODO: comment out
def run_training():
    # 1. Load your processed data
    df = pd.read_csv("data/processed/cleaned_marketing_data.csv")
    
    # 2. Scale
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend', 'Trade_Spend']
    scaler_x = MaxAbsScaler()
    scaler_y = MaxAbsScaler()
    
    X = scaler_x.fit_transform(df[channels])
    y = scaler_y.fit_transform(df[['Sales_Value']]).flatten()
    
    # 3. Build & Train
    print("Building model...")
    model = build_mmm(pd.DataFrame(X, columns=channels), y)
    
    print("Starting Sampler (this may take a few minutes)...")
    with model:
        trace = pm.sample(draws=1000, tune=1000, target_accept=0.9)
    
    # 4. Save the result
    az.to_netcdf(trace, "models/mmm_trace.nc")
    print("Success! Model saved to models/mmm_trace.nc")
    
def run_mmm_training(df):
    # 1. Initialize the model with high-level parameters
    mmm = MMM(
        date_column="Week",
        channel_columns=['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend'],
        adstock=GeometricAdstock(l_max=8), # Automates the carryover math
        saturation=LogisticSaturation(),  # Automates the diminishing returns math
        yearly_seasonality=2,             # Optional: Handles yearly spikes automatically
    )

    # 2. Fit the model
    X = df[['Week', 'TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']]
    y = df['Sales_Value']

    mmm.fit(X, y, target_accept=0.9, draws=1000, tune=1000)

    # 3. View the "built-in" results immediately
    mmm.plot_components_contributions()
    mmm.plot_channel_contribution_share_hdi()


    mmm.save("models/mmm_model_v1.nc")
    print("Model saved successfully!")

    loaded_mmm = MMM.load("models/mmm_model_v1.nc")

    print(loaded_mmm.idata.posterior.to_dataframe().head())

    loaded_mmm.plot_components_contributions()

    #plt.show()



    return mmm


if __name__ == "__main__":
    file_path = "data/processed/cleaned_marketing_data.csv" 
    m = pd.read_csv(file_path)

    m['Week'] = pd.to_datetime(m['Week'])

    print(f"Rows before de-duplication: {len(m)}")
    
    m = m.groupby('Week').agg({
        'TV_Spend': 'sum',
        'YouTube_Spend': 'sum',
        'Facebook_Spend': 'sum',
        'Instagram_Spend': 'sum',
        'Sales_Value': 'sum'
    }).reset_index()
    
    print(f"Rows after de-duplication: {len(m)}")
    
    # 3. Sort by week to ensure time-series order
    m = m.sort_values('Week')

    run_mmm_training(m)