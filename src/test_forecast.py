import pandas as pd
import numpy as np
from forecaster import run_prophet_forecast, plot_forecast

from sklearn.metrics import mean_absolute_percentage_error

# 1. Create dummy data (Weekly sales with a growth trend)
dates = pd.date_range(start="2023-01-01", periods=104, freq="W")
# Trend + Weekly Seasonality + Random Noise
y = np.linspace(100, 200, 104) + (np.sin(np.linspace(0, 10, 104)) * 20) + np.random.normal(0, 5, 104)

test_df = pd.DataFrame({'ds': dates, 'y': y})

# 2. Run the forecast
try:
    model, forecast = run_prophet_forecast(test_df, periods=12)
    print("✅ Model training and prediction successful!")
    
    # 3. Test plotting
    fig = plot_forecast(model, forecast)
    fig.show() # This should open a window with your forecast
    print("✅ Plotting successful!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")




# Split data
train = test_df.iloc[:-8]
test = test_df.iloc[-8:]

# Train and predict
model, forecast = run_prophet_forecast(train, periods=8)

# Merge actuals and predictions to compare
performance = pd.merge(test, forecast[['ds', 'yhat']], on='ds')
mape = mean_absolute_percentage_error(performance['y'], performance['yhat'])

print(f"📈 Model MAPE: {mape:.2%}") 
# Tip: For business data, a MAPE under 10-15% is usually considered excellent!