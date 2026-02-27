import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper

# --- PAGE CONFIG ---
st.set_page_config(page_title="OptiSpend AI Optimizer", layout="wide")

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_mmm_model():
    return MMM.load("models/mmm_model_v1_multi.nc")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_marketing_data.csv")
    df["Week"] = pd.to_datetime(df["Week"])
    return df

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Optimization Controls")

total_budget = st.sidebar.slider(
    "Total Weekly Budget ($)", 
    min_value=50000, 
    max_value=2000000, 
    value=1000000, 
    step=50000
)

st.sidebar.subheader("Bounds Settings")
global_lower = st.sidebar.number_input("Min Spend per Cell ($)", value=1000)
global_upper = st.sidebar.number_input("Max Spend per Cell ($)", value=60000)

# --- MAIN APP ---
st.title("🚀 OptiSpend: AI Media Allocation")
st.markdown("Optimize your spend across **Geos** and **Channels** using Bayesian MMM.")

mmm = load_mmm_model()
data = load_data()
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]
geo_list = sorted(data["Geo"].unique())

# --- RUN OPTIMIZATION ---
if st.sidebar.button("⚡ Run Optimization"):
    with st.spinner("Finding optimal allocation..."):
        
        # 1. Setup Wrapper (Simplified for App)
        start = data["Week"].max() + pd.Timedelta(weeks=1)
        end = start + pd.Timedelta(weeks=8)
        wrapper = MultiDimensionalBudgetOptimizerWrapper(mmm, start_date=start, end_date=end)
        
        # 2. Setup Bounds
        num_geos, num_channels = len(geo_list), len(channels)
        bounds_values = np.zeros((num_geos, num_channels, 2))
        bounds_values[:, :, 0] = global_lower
        bounds_values[:, :, 1] = global_upper
        
        bounds_da = xr.DataArray(
            bounds_values,
            dims=["Geo", "channel", "bound"],
            coords={"Geo": geo_list, "channel": channels, "bound": ["lower", "upper"]}
        )
        
        # Targeted Override (Matching your previous logic)
        bounds_da.loc[{"Geo": "NORTH", "channel": "Facebook_Spend", "bound": "upper"}] = 60000.0
        bounds_da.loc[{"Geo": "WEST", "channel": "TV_Spend", "bound": "upper"}] = 60000.0

        # 3. Optimize (Using your verified balanced_utility logic)
        beta_values = mmm.idata.posterior["saturation_beta"].mean(("chain", "draw")).values
        
        def app_utility(samples, budgets, **kwargs):
            import pytensor.tensor as pt
            beta_tensor = pt.as_tensor_variable(beta_values)
            return pt.mean(samples.sum()) + (budgets * beta_tensor).sum() * 1e-4

        optimal, res = wrapper.optimize_budget(
            budget=float(total_budget),
            budget_bounds=bounds_da,
            response_variable="total_media_contribution_original_scale",
            utility_function=app_utility,
        )

        # --- RESULTS LAYOUT ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Recommended Allocation")
            st.dataframe(optimal.to_dataframe(name="Spend").unstack(level=1).style.format("${:,.0f}"))

        with col2:
            st.subheader("📈 Strategy Comparison")
            # Calculate BAU
            hist_totals = data.groupby("Geo")[channels].sum()
            hist_mix = hist_totals / hist_totals.values.sum()
            bau_values = hist_mix.values * total_budget
            
            # Simple Plot
            opt_df = optimal.to_dataframe(name="Spend").reset_index()
            opt_df["Scenario"] = "AI Optimal"
            bau_df = pd.DataFrame(bau_values, index=hist_mix.index, columns=channels).stack().reset_index()
            bau_df.columns = ["Geo", "channel", "Spend"]
            bau_df["Scenario"] = "Historical Mix"
            
            comparison_df = pd.concat([opt_df, bau_df])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=comparison_df, x="Geo", y="Spend", hue="Scenario", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # --- EXPORT ---
        csv = optimal.to_dataframe(name="Weekly_Spend").to_csv().encode('utf-8')
        st.download_button("📂 Download Media Brief (CSV)", data=csv, file_name="optispend_brief.csv")

else:
    st.info("Adjust the settings on the left and click 'Run Optimization' to begin.")