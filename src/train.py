import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import MaxAbsScaler
from mmm_model import build_mmm  # Import your logic

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

if __name__ == "__main__":
    run_training()