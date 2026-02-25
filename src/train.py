import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import MaxAbsScaler
from mmm_model import build_mmm 
#from pymc_marketing.mmm.multidimensional import MMM
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

    #preprocessing and scaling
    #scaler = MaxAbsScaler()
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
    # df_scaled = df.copy()
    # df_scaled[channels] = scaler.fit_transform(df[channels])
    # df_scaled['Sales_Value'] = scaler.fit_transform(df[['Sales_Value']])


    # 1. Initialize the model with high-level parameters
    mmm = MMM(
        date_column="Week",
        channel_columns=channels,
        adstock=GeometricAdstock(l_max=8), 
        saturation=LogisticSaturation(),  
        yearly_seasonality=1, 

        scaling={
        "channel": {"method": "max", "dims": []},
        "target": {"method": "max", "dims": []},
        },

         model_config = {
        # 1. Tighten the baseline. Tell the model sales start near the average of your data.
        "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.1}},
        
        # 2. Tighten the 'Noise'. This forces the blue line to hug the black dots 
        # instead of floating in a giant cloud of uncertainty.
        "likelihood": {"dist": "Normal", "kwargs": {"sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 0.05}}}},
        
        # 3. Reduce seasonality's power so it doesn't create those giant ghost waves.
        "yearly_seasonality": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.05}},
}            
    )

    # 2. Fit the model
    # X = df_scaled[['Week'] + channels]
    # y_scaled = df_scaled['Sales_Value']

    X = df[['Week'] + channels]
    y = df['Sales_Value'].rename("y")

    mmm.fit(X, y, target_accept=0.9, draws=1000, tune=1000)

    # 3. View the "built-in" results immediately
    # mmm.plot_components_contributions()
    # mmm.plot_channel_contribution_share_hdi()


    mmm.save("models/mmm_model_v1.nc")
    print("Model saved successfully!")

    # loaded_mmm = MMM.load("models/mmm_model_v1.nc")

    # # print(loaded_mmm.idata.posterior.to_dataframe().head())

    # loaded_mmm.plot_components_contributions()

    # #plt.show()
    # plt.savefig("plots/components_contribution.png")

    return mmm


def run_mmm_training(df):

    #preprocessing and scaling
    scaler = MaxAbsScaler()
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
    df_scaled = df.copy()
    df_scaled[channels] = scaler.fit_transform(df[channels])
    df_scaled['Sales_Value'] = scaler.fit_transform(df[['Sales_Value']])


    # 1. Initialize the model with high-level parameters
    mmm = MMM(
        date_column="Week",
        channel_columns=channels,
        adstock=GeometricAdstock(l_max=8), 
        saturation=LogisticSaturation(),  
        yearly_seasonality=1, 

         model_config = {
        # 1. Tighten the baseline. Tell the model sales start near the average of your data.
        "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.1}},
        
        # 2. Tighten the 'Noise'. This forces the blue line to hug the black dots 
        # instead of floating in a giant cloud of uncertainty.
        "likelihood": {"dist": "Normal", "kwargs": {"sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 0.05}}}},
        
        # 3. Reduce seasonality's power so it doesn't create those giant ghost waves.
        "yearly_seasonality": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.05}},
}            
    )

    # 2. Fit the model
    X = df_scaled[['Week'] + channels]
    y_scaled = df_scaled['Sales_Value']

    # X = df[['Week'] + channels]
    # y = df['Sales_Value'].rename("y")

    mmm.fit(X, y_scaled, target_accept=0.9, draws=1000, tune=1000)

    # 3. View the "built-in" results immediately
    # mmm.plot_components_contributions()
    # mmm.plot_channel_contribution_share_hdi()


    mmm.save("models/mmm_model_v1.nc")
    print("Model saved successfully!")

    # loaded_mmm = MMM.load("models/mmm_model_v1.nc")

    # # print(loaded_mmm.idata.posterior.to_dataframe().head())

    # loaded_mmm.plot_components_contributions()

    # #plt.show()
    # plt.savefig("plots/components_contribution.png")

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