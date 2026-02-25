import pandas as pd
df_raw = pd.read_csv("data/processed/cleaned_marketing_data.csv")

print(f"ACTUAL Total Sales in CSV:  ${df_raw['Sales_Value'].sum():,.2f}")
print(f"ACTUAL Total Spend in CSV:  ${(df_raw['TV_Spend'].sum() + df_raw['YouTube_Spend'].sum() + df_raw['Facebook_Spend'].sum() + df_raw['Instagram_Spend'].sum()):,.2f}")