import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import MaxAbsScaler
from mmm_model import build_mmm 
from pymc_marketing.mmm import MMM
#from pymc_marketing.mmm.multidimensional import MMM
import matplotlib.pyplot as plt
import os
import arviz as az


# 1. Load the data
data = pd.read_csv("data/processed/cleaned_marketing_data.csv")
data['Week'] = pd.to_datetime(data['Week'])

# 2. IMPORTANT: Re-apply the same scaling used in training
scaler_x = MaxAbsScaler()
scaler_y = MaxAbsScaler()

channels = ['TV_Spend', 'YouTube_Spend', 'Facebook_Spend', 'Instagram_Spend']
data_scaled = data.copy()
data_scaled[channels] = scaler_x.fit_transform(data[channels])
data_scaled['Sales_Value'] = scaler_y.fit_transform(data[['Sales_Value']])

# 3. Load the model
loaded_mmm = MMM.load("models/mmm_model_v1.nc")


X_to_predict = data_scaled.drop(['Sales_Value'], axis=1)
y_to_match = data_scaled['Sales_Value']

# 2. OVERWRITE self.y right before predicting/plotting
# The model needs this to compare 'Actual' vs 'Predicted'
loaded_mmm.y = y_to_match

# 3. Generate predictions
print("Generating predictions...")
loaded_mmm.sample_posterior_predictive(X_to_predict, combined=True)

# 4. Plot
print("Plotting model fit...")
# Since you used manual scaling, set original_scale=False to see the 0-1 fit
loaded_mmm.plot_posterior_predictive(original_scale=False)
plt.title("Model Fit: Scaled Actual vs Predicted")
plt.savefig("plots/model_fit_check.png")