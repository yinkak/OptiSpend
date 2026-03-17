import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper
from src.forecaster import get_prophet_ready_data, run_prophet_forecast, plot_forecast

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="OptiSpend AI Optimizer", layout="wide")

# --- 2. CACHED DATA LOADING ---
@st.cache_resource
def load_mmm_model():
    return MMM.load("models/mmm_model_v1_multi.nc")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_marketing_data.csv")
    df["Week"] = pd.to_datetime(df["Week"])
    return df

# Load assets
mmm = load_mmm_model()
data = load_data()
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]
geo_list = sorted(data["Geo"].unique())

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Optimization Controls")

# Initialize session state for budget if it doesn't exist
if 'total_budget' not in st.session_state:
    st.session_state.total_budget = 1000000

# 1. Manual Number Input
manual_budget = st.sidebar.number_input(
    "Total Weekly Budget ($) - Manual Input",
    min_value=50000,
    max_value=2000000,
    value=st.session_state.total_budget,
    step=10000,
    key="budget_input"
)

# 2. Slider Input (Linked to the same value)
# We use st.session_state.total_budget to sync them
total_budget = st.sidebar.slider(
    "Total Weekly Budget ($) - Slider", 
    min_value=50000, 
    max_value=2000000, 
    value=manual_budget, # Forces slider to follow the manual input
    step=50000
)

# Update the state so the rest of the app uses 'total_budget'
st.session_state.total_budget = total_budget

st.sidebar.subheader("Bounds Settings")
global_lower = st.sidebar.number_input("Min Spend per Cell ($)", value=1000)
global_upper = st.sidebar.number_input("Max Spend per Cell ($)", value=60000)

# --- 4. TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["🎯 Budget Optimization", "📈 Sales Forecasting", "📊 Model Health"])

# --- TAB 1: OPTIMIZATION ---
with tab1:
    st.title("🚀 OptiSpend: AI Media Allocation")
    st.markdown("Optimize spend across **Geos** and **Channels** using Bayesian MMM.")

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
            
            # Targeted Override
            bounds_da.loc[{"Geo": "NORTH", "channel": "Facebook_Spend", "bound": "upper"}] = 60000.0
            bounds_da.loc[{"Geo": "WEST", "channel": "TV_Spend", "bound": "upper"}] = 60000.0

            # 3. Optimize
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
        st.info("Adjust the settings on the left and click 'Run Optimization'.")

# --- TAB 2: FORECASTING ---
with tab2:
    st.header("📈 Organic Baseline Forecast")
    st.write("Predicting future sales by stripping away marketing impact to find the brand's 'true' pulse.")
    
    if st.button("🔮 Generate 12-Week Forecast"):
        with st.spinner("Decomposing baseline and running Prophet..."):
            prophet_df = get_prophet_ready_data(mmm, data)
            model, forecast = run_prophet_forecast(prophet_df, periods=12)
            
            # Show Plot
            fig = plot_forecast(model, forecast)
            st.pyplot(fig)
            
            # Show Components (Trend/Seasonality)
            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

# --- TAB 3: MODEL HEALTH ---
with tab3:
    st.header("📊 Saturation & ROI")
    
    # 1. Extract Contribution for Pie Chart
    # We pull 'channel_contribution' from the posterior and take the median
    if "channel_contribution" in mmm.idata.posterior:
        # Summing across 'chain', 'draw', 'date', and 'Geo' to get total per channel
        total_contrib = mmm.idata.posterior["channel_contribution"].sum(dim=["date", "Geo"]).median(dim=["chain", "draw"])
        contrib_df = total_contrib.to_series()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Media Contribution Share")
            fig3, ax3 = plt.subplots()
            contrib_df.index = [c.replace('"', '') for c in contrib_df.index]
            contrib_df.plot.pie(autopct='%1.1f%%', ax=ax3, legend=False)
            ax3.set_ylabel("")
            st.pyplot(fig3)
            
        with col2:
            st.subheader("Channel Effectiveness (Betas)")
            # Pull the 'saturation_beta' which represents the strength of each channel
            betas = mmm.idata.posterior["saturation_beta"].mean(dim=["chain", "draw", "Geo"]).to_series()
            betas.index = [c.replace('"', '') for c in betas.index]
            st.bar_chart(betas)
    else:
        st.warning("Channel contribution data not found in model. Try re-running the optimization to populate results.")