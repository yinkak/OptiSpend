import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error




def get_prophet_ready_data(mmm, df):
    """
    Extracts the organic baseline from a MultiDimensional MMM and formats it for Prophet.
    """
    # 1. Pull the raw channel contribution (chain, draw, date, Geo, channel)
    posterior_contrib = mmm.idata.posterior["channel_contribution"].median(dim=["chain", "draw"])
    
    # 2. Sum across all channels to get total impact per (Date, Geo)
    total_media_impact = posterior_contrib.sum(dim="channel")
    
    # 3. Convert Xarray to a clean Pandas DataFrame for easy merging
    impact_df = total_media_impact.to_dataframe(name="media_impact").reset_index()
    

    if 'date' in impact_df.columns:
        impact_df = impact_df.rename(columns={'date': 'Week'})
        
    merged_df = pd.merge(df, impact_df, on=['Week', 'Geo'], how='left')
    
    # 5. Calculate the Baseline: Actual Sales - Media Impact
    merged_df['organic_baseline'] = merged_df['Sales_Value'] - merged_df['media_impact']
    
    # 6. Aggregate to National Level for Prophet
    national_baseline = merged_df.groupby('Week')['organic_baseline'].sum().reset_index()
    
    prophet_df = national_baseline.rename(columns={'Week': 'ds', 'organic_baseline': 'y'})
    
    return prophet_df

def run_prophet_forecast(prophet_df, periods=12):
    """
    Step 2: Train the Prophet model and project into the future.
    """
    # Initialize the model with yearly seasonality enabled
    # This helps Prophet understand 'Summer vs Winter' patterns
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False
    )
    
    # Fit the model to our historical baseline
    model.fit(prophet_df)
    
    # Create a 'Future Dataframe' extending X weeks out
    future = model.make_future_dataframe(periods=periods, freq='W')
    
    # Predict!
    # This returns 'yhat' (prediction), 'yhat_lower', and 'yhat_upper'
    forecast = model.predict(future)
    
    return model, forecast

def plot_forecast(model, forecast):
    """
    Step 3: Create a clean visualization.
    """
    fig = model.plot(forecast)
    plt.title("Organic Sales Forecast (12-Week Outlook)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Baseline Sales")
    plt.savefig("plots/forecast_test.png")
    return fig