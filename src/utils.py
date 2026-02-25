# src/utils.py
import joblib
import numpy as np

def unscale_sales(scaled_sales_data):
    scaler = joblib.load("models/salesscaler.joblib")
    
    # Convert to numpy array if it's a pandas/xarray scalar
    data_array = np.array(scaled_sales_data)
    
    # Force it into a 2D shape (1 row, 1 column)
    reshaped_data = data_array.reshape(-1, 1)
    
    return scaler.inverse_transform(reshaped_data)

def unscale_spend(scaled_spend_data):
    """
    Translates 0-1 spend back into real-world costs.
    """
    scaler = joblib.load("models/spendscaler.joblib")
    data_array = np.array(scaled_spend_data)
    
    # If passing a single channel's mean, reshape to 2D
    if data_array.ndim == 0 or data_array.size == 1:
        reshaped_data = data_array.reshape(-1, 1)
        # Note: scaler.inverse_transform expects 4 columns if it was fit on 4
        # We handle this by using the scaler's transform logic or 
        # just manually multiplying by the specific channel's max if needed.
        # For simplicity, if you're unscaling the whole table:
        return scaler.inverse_transform(reshaped_data)
    
    return scaler.inverse_transform(data_array)


