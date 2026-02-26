import pandas as pd
import joblib
from sklearn.preprocessing import MaxAbsScaler
# Import the specific Multidimensional class
from pymc_marketing.mmm.multidimensional import MMM 
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation

def run_mmm_multidimensional_training(df):
    # 1. Preprocessing and Scaling
    # MDMMM handles data with multiple rows per date, so we scale the whole column
    spend_scaler = MaxAbsScaler()
    sales_scaler = MaxAbsScaler()
    channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
    
    df_scaled = df.copy()
    df_scaled[channels] = spend_scaler.fit_transform(df[channels])
    df_scaled['Sales_Value'] = sales_scaler.fit_transform(df[['Sales_Value']])

    # 2. Initialize the Multidimensional Model
    # 'dims' is the key change here. It tells the model to treat 'Geo' as a dimension.
    mmm = MMM(
        date_column="Week",
        channel_columns=channels,
        dims=("Geo",),  # <--- CRITICAL: Tells the model to look at Geography
        target_column="Sales_Value", # <--- Required for MDMMM
        adstock=GeometricAdstock(l_max=8), 
        saturation=LogisticSaturation(),  
        yearly_seasonality=1, 
        model_config={
            "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.1}},
            "likelihood": {"dist": "Normal", "kwargs": {"sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 0.05}}}},
            "yearly_seasonality": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 0.05}},
            "saturation_beta": {
                "dist": "HalfNormal",
                "kwargs": {"sigma": 2},
                "dims": ("Geo", "channel"),
            }
        }            
    )

    # 3. Fit the model
    # Note: We do NOT groupby('Week') anymore; the model uses (Week + Geo) as the index
    X = df_scaled[['Week', 'Geo'] + channels]
    y_scaled = df_scaled['Sales_Value']

    print("Starting Multidimensional Training...")
    mmm.fit(X, y_scaled, target_accept=0.9, draws=1000, tune=1000)

    # 4. Save Assets
    joblib.dump(spend_scaler, "models/spendscaler_multidim.joblib")
    joblib.dump(sales_scaler, "models/salesscaler_multidim.joblib")
    mmm.save("models/mmm_model_v1_multi.nc")
    
    print("Multidimensional Model saved successfully!")
    return mmm

if __name__ == "__main__":
    file_path = "data/processed/cleaned_marketing_data.csv" 
    m = pd.read_csv(file_path)
    m['Week'] = pd.to_datetime(m['Week'])

    # IMPORTANT: We no longer aggregate by Week. 
    # We keep the 'Geo' column so the model learns from each region.
    # We only aggregate to ensure there's only 1 entry per Week per Geo.
    m = m.groupby(['Week', 'Geo']).agg({
        'TV_Spend': 'sum',
        'YouTube_Spend': 'sum',
        'Facebook_Spend': 'sum',
        'Instagram_Spend': 'sum',
        'Sales_Value': 'sum'
    }).reset_index()
    
    m = m.sort_values(['Week', 'Geo'])

    run_mmm_multidimensional_training(m)