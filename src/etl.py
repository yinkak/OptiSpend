from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load raw data
df_raw = pd.read_csv("data/processed/cleaned_marketing_data.csv")

df_numeric = df_raw.drop(columns=['Week'], errors='ignore')
# Convert 'Geo' to numbers so the math works
df_numeric = pd.get_dummies(df_numeric)

# Define features (all your columns except Sales and Week)
cols_to_drop = ['Sales_Value', 'Week', 'log_sales_val', 'log_sales'] # add any other sales-related columns
X = df_numeric.drop(columns=[c for c in cols_to_drop if c in df_numeric.columns])
y = df_numeric['Sales_Value']


# Train a quick importance model
rf = RandomForestRegressor().fit(X, y)

# Print Importance
importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
print(importance.sort_values(by='Importance', ascending=False))