# 🎯 OptiSpend: Bayesian Marketing Mix Modeling & Optimization

OptiSpend is an end-to-end marketing analytics suite designed to quantify the impact of media spend across multiple geographies and channels. Using Bayesian Structural Time Series and Causal Inference, it transforms raw marketing data into actionable budget reallocations.

## 🚀 Key Features

* **Bayesian MMM:** A hierarchical model that estimates saturation and decay (adstock) effects of media spend.
* **Budget Optimizer:** Uses a custom utility function to find the "Maximum Yield" budget allocation across regions.
* **Forecasting Lab:** Integrated with Facebook Prophet to predict baseline sales trends and future acquisition.
* **Causal Inference:** An experimentation module using Google's `CausalImpact` to measure incremental lift from specific marketing interventions.

## 🛠️ Technical Stack

* **Language:** Python 3.12
* **Modeling:** `PyMC` (Bayesian Inference), `Prophet` (Time-Series Forecasting)
* **Causal Analysis:** `CausalImpact` (Bayesian Structural Time Series)
* **Frontend:** `Streamlit` (Interactive Dashboard)
* **Optimization:** `Scipy.optimize` (Sequential Least Squares Programming)

## 📊 How it Works

### 1. The Model (Marketing Mix Modeling)
The core engine uses a Logistic Saturation function to model diminishing returns.   

$$Contribution = \beta \cdot \frac{1 - e^{-\alpha \cdot spend}}{1 + e^{-\alpha \cdot spend}}$$


It accounts for geography-specific nuances through hierarchical priors, allowing the model to "learn" from one region to inform another.

## 📂 Project Structure

```text
OptiSpend/
├── app/
│   └── main.py            
├── data/
│   ├── raw/               
│   └── processed/         
├── models/                
├── src/
│   ├── train.py           
│   ├── causal_analysis.py 
│   └── utils.py           
├── reports/               
├── requirements.txt
└── README.md
```

### 2. The Experimentation Lab
Unlike traditional models, OptiSpend includes a Causal Inference module. By defining a "Pre-Period" and "Post-Period," the system builds a synthetic control group to isolate the **true incremental lift** of a campaign, filtering out seasonality and market noise.

## 🏃 Run it Locally

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/OptiSpend.git](https://github.com/your-username/OptiSpend.git)
   cd OptiSpend
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app:**
    ```bash
    streamlit run app/main.py
    ```

## 📈 Roadmap

- [x] **Phase 1-3: Core MMM Engine & Optimizer**
  - Bayesian hierarchical modeling with PyMC.
  - Custom utility functions for budget reallocation.
  - Interactive Streamlit dashboard for ROI visualization.
- [x] **Phase 4: Forecasting & Experimentation**
  - Integrated Prophet modules for baseline sales projection.
  - CausalImpact implementation for incremental lift analysis.
  - Automated data-snapping for weekly time-series alignment.
- [ ] **Phase 5: Cloud Deployment & Scalability**
  - Deploy to Streamlit Cloud for live stakeholder access.
  - Optimize model pickling for faster load times.
  - Add API endpoints for automated weekly data ingestion.


### 📦 Model Storage & GitHub Constraints

The core model output (`mmm_trace.nc`) is a high-dimensional Bayesian trace file approximately **538 MB** in size. 

**Why it is not in this repository:**
* **GitHub File Limits:** GitHub enforces a 100 MB limit per file. 
* **Git Integrity:** Large binary files bloat repository size and slow down cloning for other users.
* **Reproducibility:** In line with ML best practices, this project prioritizes **Code-as-Infrastructure**. Instead of downloading a static model, users are encouraged to generate the trace locally to ensure environment compatibility.

**How to generate the model locally:**
If you wish to perform analysis or run the dashboard, you must generate the trace file on your local machine:
1. Ensure your `venv` is active.
2. Run the training pipeline:
   ```bash
   python3 src/train.py
   ```
3. The script will output the mmm_trace.nc file into the models/ directory, which is ignored by Git.


