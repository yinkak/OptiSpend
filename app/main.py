import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pymc_marketing.mmm.multidimensional import MMM, MultiDimensionalBudgetOptimizerWrapper
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecaster import get_prophet_ready_data, run_prophet_forecast, plot_forecast

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="OptiSpend AI Optimizer", layout="wide")

# --- 2. CONSOLIDATED DATA & MODEL LOADING ---

@st.cache_resource
def load_mmm_model():
    model_path = os.path.join(os.getcwd(), "models", "mmm_model_v1_multi.nc")
    if os.path.exists(model_path):
        try:
            return MMM.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

@st.cache_data
def load_data():
    # Adding a check for the data file too, just in case!
    data_path = "data/processed/cleaned_marketing_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["Week"] = pd.to_datetime(df["Week"])
        return df
    return pd.DataFrame() # Return empty df if missing

# EXECUTE LOADING ONCE
mmm = load_mmm_model()
data = load_data()

# Identify if we are in Demo Mode
is_demo_mode = mmm is None

# Define shared variables
channels = ["TV_Spend", "YouTube_Spend", "Facebook_Spend", "Instagram_Spend"]
geo_list = sorted(data["Geo"].unique()) if not data.empty else ["NORTH", "SOUTH", "EAST", "WEST"]

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
tab1, tab2, tab3, tab4= st.tabs(["🎯 Budget Optimization", "📈 Sales Forecasting", "📊 Model Health", "🧪 Experimentation"])

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
    if "channel_contribution" in mmm.idata.posterior:
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
            betas = mmm.idata.posterior["saturation_beta"].mean(dim=["chain", "draw", "Geo"]).to_series()
            betas.index = [c.replace('"', '') for c in betas.index]
            st.bar_chart(betas)
    else:
        st.warning("Channel contribution data not found in model. Try re-running the optimization to populate results.")



with tab4:
    st.header("🧪 Incrementality Test (Causal Inference)")
    st.write("Determine the true 'lift' of a specific marketing event by comparing it to synthetic control regions.")

    col1, col2 = st.columns(2)
    with col1:
        target_geo = st.selectbox("Select Target Region for Test", options=data['Geo'].unique())
        test_start = st.date_input("Campaign Start Date", value=pd.to_datetime("2023-11-04"))
        
    with col2:
        control_geos = [g for g in data['Geo'].unique() if g != target_geo]
        st.write(f"**Control Regions:** {', '.join(control_geos)}")
        test_end = st.date_input("Campaign End Date", value=pd.to_datetime("2024-01-13"))

    if st.button("Analyze Campaign Impact"):
        with st.spinner("Building Bayesian Structural Time Series..."):
            # 1. Prepare Data
            df_agg = data.groupby(['Week', 'Geo'])['Sales_Value'].sum().reset_index()
            analysis_df = df_agg.pivot(index='Week', columns='Geo', values='Sales_Value')
            
            # Reorder: Target first, then controls
            cols = [target_geo] + control_geos
            analysis_df = analysis_df[cols]
            
            # 2. Define Periods
            available_dates = analysis_df.index

            test_start_ts = pd.Timestamp(test_start)
            test_end_ts = pd.Timestamp(test_end)

            pre_end = available_dates[available_dates < test_start_ts].max()
            pre_start = available_dates.min()

            post_start = available_dates[available_dates >= test_start_ts].min()
            post_end = available_dates[available_dates <= test_end_ts].max()

            if pd.isnull(post_start) or pd.isnull(post_end):
                st.error("Selected dates fall outside the range of the dataset. Please adjust.")
            else:
                pre_period = [str(pre_start.date()), str(pre_end.date())]
                post_period = [str(post_start.date()), str(post_end.date())]

            
            # 3. Run CausalImpact
            from causalimpact import CausalImpact
            ci = CausalImpact(analysis_df, pre_period, post_period)
            
            # 4. Display Results
            st.subheader("Results Summary")
            try:
                # Accessing the relative effect from the summary table
                lift_val = ci.summary_data.loc['rel_effect', 'average']
                lift = lift_val * 100
                prob = 1 - ci.p_value # Confidence
            except Exception as e:
                lift = 0.0
                prob = 0.0
                st.error(f"Data mapping error: {e}")
                        
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Relative Lift", f"{lift:.2f}%")
            stat_col2.metric("Stat. Confidence", f"{(1-prob)*100:.1f}%")
            
            # 5. Show the Plot
            fig = ci.plot()
            st.pyplot(fig)
            
            # 6. Professional Interpretation
            st.info(f"**Insight:** The campaign in {target_geo} resulted in a "
                    f"{'positive' if lift > 0 else 'negative'} incremental impact. "
                    f"The probability of this being a fluke is {prob*100:.2f}%.")